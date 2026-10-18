"""
Multi-Fidelity Closed-Loop Active Learning & Bayesian Optimization for PolymerGenesis.
Fuses Low-Fidelity GNN surrogate predictions (s=0.0) with High-Fidelity Molecular Dynamics (s=1.0).
Uses BoTorch SingleTaskMultiFidelityGP to minimize computational cost while guaranteeing physics verification.
"""

import torch
import numpy as np
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction

class MultiFidelityPolymerOptimizer:
    def __init__(self, latent_dim: int = 64, bounds_range: float = 3.0):
        self.latent_dim = latent_dim
        self.bounds_range = bounds_range
        # Unit hypercube bounds [0.0, 1.0] for D latent dimensions + 1 fidelity dimension
        self.unit_bounds = torch.tensor([
            [0.0] * (latent_dim + 1),
            [1.0] * (latent_dim + 1)
        ], dtype=torch.double)
        self.fidelity_dim = latent_dim # index 64

    def _scale_X(self, train_X: torch.Tensor) -> torch.Tensor:
        """Min-max scales latent coordinates to [0, 1], leaving fidelity [0, 1] untouched."""
        X_d = train_X.clone().detach().to(dtype=torch.double)
        latent_norm = (X_d[:, :self.latent_dim] + self.bounds_range) / (2.0 * self.bounds_range)
        latent_norm = torch.clamp(latent_norm, 0.0, 1.0)
        fid = torch.clamp(X_d[:, self.fidelity_dim:], 0.0, 1.0)
        return torch.cat([latent_norm, fid], dim=-1)

    def _unscale_latent(self, candidates_norm: torch.Tensor) -> torch.Tensor:
        """Unscales normalized latent coordinates back to [-bounds_range, bounds_range]."""
        return candidates_norm * (2.0 * self.bounds_range) - self.bounds_range

    def fit_gp(self, train_X: torch.Tensor, train_Y: torch.Tensor):
        """
        Fits SingleTaskMultiFidelityGP on the unit hypercube.
        """
        train_X_norm = self._scale_X(train_X)
        train_Y_d = train_Y.clone().detach().to(dtype=torch.double)

        y_mean = train_Y_d.mean()
        y_std = train_Y_d.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1.0, dtype=torch.double)
        norm_Y = (train_Y_d - y_mean) / y_std

        gp = SingleTaskMultiFidelityGP(
            train_X=train_X_norm,
            train_Y=norm_Y,
            data_fidelities=[self.fidelity_dim]
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
        return gp, y_mean, y_std

    def suggest_candidates(self, train_X: torch.Tensor, train_Y: torch.Tensor, q: int = 5, beta: float = 2.0, target_fidelity: float = 1.0):
        """
        Optimizes acquisition function conditioned on target high-fidelity (s=1.0)
        to discover candidates for high-fidelity physics simulation.
        """
        gp, y_mean, y_std = self.fit_gp(train_X, train_Y)

        # Acquisition function conditioned on high fidelity (s=1.0)
        qucb = qUpperConfidenceBound(gp, beta=beta)
        fixed_acq = FixedFeatureAcquisitionFunction(
            acq_function=qucb,
            d=self.latent_dim + 1,
            columns=[self.fidelity_dim],
            values=[target_fidelity]
        )

        # Optimize over the D latent dimensions within [0.0, 1.0]^D
        latent_unit_bounds = self.unit_bounds[:, :self.latent_dim].to(train_X.device)
        candidates_latent_norm, _ = optimize_acqf(
            acq_function=fixed_acq,
            bounds=latent_unit_bounds,
            q=q,
            num_restarts=4,
            raw_samples=16
        )

        # Unscale latent coordinates back to original continuous space
        candidates_latent = self._unscale_latent(candidates_latent_norm)

        # Append target fidelity to returned candidates
        fid_col = torch.full((q, 1), target_fidelity, dtype=torch.float, device=train_X.device)
        full_candidates = torch.cat([candidates_latent.float(), fid_col], dim=-1)

        return full_candidates
