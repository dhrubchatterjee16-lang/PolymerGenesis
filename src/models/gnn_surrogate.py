"""
Graph Neural Network Surrogate Model with Monte Carlo Dropout (MCD) for Uncertainty Quantification.
Architecture:
- Multi-layer Message Passing (GCN / GraphConv) with residual skip connections
- Dual-Pooling: global_mean_pool + global_max_pool
- Epistemic Uncertainty via Test-Time Monte Carlo Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

class PolymerGNN(nn.Module):
    def __init__(self, in_channels: int = 30, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.2, edge_dim: int = 6):
        super(PolymerGNN, self).__init__()
        self.dropout_rate = dropout
        self.num_layers = num_layers
        self.edge_dim = edge_dim

        # Node feature projection
        self.node_proj = nn.Linear(in_channels, hidden_dim)

        # Message passing layers with bond attributes (GINEConv)
        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ),
                edge_dim=edge_dim
            ) for _ in range(num_layers)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Dual pooling output dimension = hidden_dim * 2
        fc_in = hidden_dim * 2
        self.fc1 = nn.Linear(fc_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        
        # Dual-head: predicts mean (mu) and log-variance (log_var) for heteroscedastic aleatoric uncertainty
        self.head_mu = nn.Linear(64, 1)
        self.head_var = nn.Linear(64, 1)

    def forward(self, x, edge_index, batch, apply_dropout: bool = False, edge_attr: torch.Tensor = None):
        h = F.relu(self.node_proj(x))

        # Handle edge attributes (if None or empty, default to zero bond features)
        if edge_attr is None or edge_attr.numel() == 0:
            num_edges = edge_index.size(1) if edge_index is not None and edge_index.dim() == 2 else 0
            edge_attr = torch.zeros((num_edges, self.edge_dim), dtype=torch.float, device=x.device)

        for i in range(self.num_layers):
            h_conv = self.convs[i](h, edge_index, edge_attr=edge_attr)
            h_conv = self.batch_norms[i](h_conv)
            h_conv = F.relu(h_conv)
            if apply_dropout:
                h_conv = F.dropout(h_conv, p=self.dropout_rate, training=True)
            # Residual connection
            h = h + h_conv

        # Global readout pooling (mean + max)
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_pool = torch.cat([h_mean, h_max], dim=-1)

        out = F.relu(self.fc1(h_pool))
        if apply_dropout:
            out = F.dropout(out, p=self.dropout_rate, training=True)
        out = F.relu(self.fc2(out))
        if apply_dropout:
            out = F.dropout(out, p=self.dropout_rate, training=True)

        mu = self.head_mu(out).squeeze(-1)
        log_var = self.head_var(out).squeeze(-1)
        return mu, log_var

    def predict_monte_carlo(self, x, edge_index, batch, num_samples: int = 50, edge_attr: torch.Tensor = None):
        """
        Runs T forward passes with active dropout to compute:
        - Predictive mean (mu)
        - Epistemic uncertainty (std_epistemic)
        - Aleatoric uncertainty (std_aleatoric)
        - Total uncertainty (std_total)
        """
        self.train() # Keep dropout active
        mus = []
        log_vars = []

        with torch.no_grad():
            for _ in range(num_samples):
                mu_sample, log_var_sample = self.forward(x, edge_index, batch, apply_dropout=True, edge_attr=edge_attr)
                mus.append(mu_sample.cpu())
                log_vars.append(log_var_sample.cpu())

        mus = torch.stack(mus)         # shape: (T, N)
        log_vars = torch.stack(log_vars) # shape: (T, N)

        # Epistemic variance (variance of predictions across dropout passes)
        mu_mean = torch.mean(mus, dim=0)
        var_epistemic = torch.var(mus, dim=0)

        # Aleatoric variance (mean of predicted variances)
        var_aleatoric = torch.mean(torch.exp(log_vars), dim=0)

        # Total uncertainty
        var_total = var_epistemic + var_aleatoric
        std_total = torch.sqrt(var_total)

        return (
            mu_mean.numpy(),
            std_total.numpy(),
            torch.sqrt(var_epistemic).numpy(),
            torch.sqrt(var_aleatoric).numpy()
        )


class PhysicsInformedPolymerGNN(PolymerGNN):
    """
    Physics-Informed Graph Neural Network with Fox-Flory Molecular Weight Conditioning.
    Enforces the fundamental thermodynamic law of polymer glass transitions:
        Tg(M_n) = Tg_inf - K / M_n
    where:
    - Tg_inf is asymptotic glass transition at infinite molecular weight (M_n -> inf).
    - K > 0 is excess free-volume parameter per chain end (strictly positive via Softplus).
    - M_n is number-average molecular weight in kDa (e.g. 50 kDa = 50,000 g/mol).
    """
    def __init__(self, in_channels: int = 30, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.2, edge_dim: int = 6):
        super().__init__(in_channels=in_channels, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, edge_dim=edge_dim)
        # Dedicated K-head predicting excess free volume parameter K > 0 (scaled in K * kDa)
        self.head_k = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() # Enforces thermodynamic positivity K > 0
        )

    def forward(self, x, edge_index, batch, apply_dropout: bool = False, m_n: torch.Tensor = None, edge_attr: torch.Tensor = None):
        h = F.relu(self.node_proj(x))

        if edge_attr is None or edge_attr.numel() == 0:
            num_edges = edge_index.size(1) if edge_index is not None and edge_index.dim() == 2 else 0
            edge_attr = torch.zeros((num_edges, self.edge_dim), dtype=torch.float, device=x.device)

        for i in range(self.num_layers):
            h_conv = self.convs[i](h, edge_index, edge_attr=edge_attr)
            h_conv = self.batch_norms[i](h_conv)
            h_conv = F.relu(h_conv)
            if apply_dropout:
                h_conv = F.dropout(h_conv, p=self.dropout_rate, training=True)
            # Residual connection
            h = h + h_conv

        # Global readout pooling (mean + max)
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_pool = torch.cat([h_mean, h_max], dim=-1)

        out = F.relu(self.fc1(h_pool))
        if apply_dropout:
            out = F.dropout(out, p=self.dropout_rate, training=True)
        out = F.relu(self.fc2(out))
        if apply_dropout:
            out = F.dropout(out, p=self.dropout_rate, training=True)

        mu_inf = self.head_mu(out).squeeze(-1)
        log_var = self.head_var(out).squeeze(-1)

        if m_n is not None:
            m_n_tensor = torch.as_tensor(m_n, dtype=torch.float, device=x.device)
            if m_n_tensor.dim() == 0:
                m_n_tensor = m_n_tensor.unsqueeze(0)
            m_n_clamped = torch.clamp(m_n_tensor, min=1.0)
            # Scale K factor (typically ~100 K * kDa)
            k_val = self.head_k(out).squeeze(-1) * 100.0
            fox_flory_reduction = k_val / m_n_clamped
            mu = mu_inf - fox_flory_reduction
        else:
            mu = mu_inf

        return mu, log_var

    def predict_monte_carlo(self, x, edge_index, batch, num_samples: int = 50, m_n: torch.Tensor = None, edge_attr: torch.Tensor = None):
        """Runs T Monte Carlo forward passes with Fox-Flory conditioning."""
        self.train()
        mus = []
        log_vars = []

        with torch.no_grad():
            for _ in range(num_samples):
                mu_sample, log_var_sample = self.forward(x, edge_index, batch, apply_dropout=True, m_n=m_n, edge_attr=edge_attr)
                mus.append(mu_sample.cpu())
                log_vars.append(log_var_sample.cpu())

        mus = torch.stack(mus)
        log_vars = torch.stack(log_vars)

        mu_mean = torch.mean(mus, dim=0)
        var_epistemic = torch.var(mus, dim=0)
        var_aleatoric = torch.mean(torch.exp(log_vars), dim=0)
        var_total = var_epistemic + var_aleatoric
        std_total = torch.sqrt(var_total)

        return (
            mu_mean.numpy(),
            std_total.numpy(),
            torch.sqrt(var_epistemic).numpy(),
            torch.sqrt(var_aleatoric).numpy()
        )


def gaussian_nll_loss(y_true, mu, log_var):
    """
    Heteroscedastic Gaussian Negative Log-Likelihood loss:
    Loss = 0.5 * exp(-log_var) * (y_true - mu)^2 + 0.5 * log_var
    """
    precision = torch.exp(-log_var)
    diff = (y_true - mu) ** 2
    return torch.mean(0.5 * precision * diff + 0.5 * log_var)

