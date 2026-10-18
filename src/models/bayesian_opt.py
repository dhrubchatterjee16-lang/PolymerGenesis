"""
Latent Space Bayesian Optimization (BoTorch) and Guided Exploration for Polymer Design.
"""

import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf

class LatentBayesianOptimizer:
    def __init__(self, latent_dim: int = 64, bounds_range: float = 3.0, use_turbo: bool = True):
        """
        Latent Space Bayesian Optimizer equipped with:
        1. Trust-Region Bayesian Optimization (TuRBO) for high-dimensional stability in D=64.
        2. Numerically robust qLogExpectedImprovement (qLogEI) and qUCB acquisition functions.
        """
        self.latent_dim = latent_dim
        self.bounds_range = bounds_range
        self.use_turbo = use_turbo

        # Standard unit hypercube bounds [0.0, 1.0]^D
        self.unit_bounds = torch.tensor([
            [0.0] * latent_dim,
            [1.0] * latent_dim
        ], dtype=torch.double)

        # TuRBO State Variables (Eriksson et al., NeurIPS 2019)
        self.turbo_length = 0.8
        self.turbo_length_min = 0.5 ** 7 # ~0.0078
        self.turbo_length_max = 1.6
        self.turbo_success_counter = 0
        self.turbo_failure_counter = 0
        self.turbo_success_tolerance = 3
        self.turbo_failure_tolerance = 4
        self.best_reward = -float('inf')

    def optimize_step(self, z_history: torch.Tensor, rewards: torch.Tensor, q: int = 5, beta: float = 2.0, acq_type: str = "qLogEI"):
        """
        Fits a Gaussian Process on min-max scaled unit hypercube and optimizes
        either qLogExpectedImprovement (qLogEI) or q-Upper Confidence Bound (qUCB).
        Applies TuRBO local trust-region bounding box to prevent 64D dimensional collapse.
        """
        # Ensure double precision and min-max scale to unit cube [0, 1]
        train_X_raw = z_history.clone().detach().to(dtype=torch.double)
        train_X_norm = (train_X_raw + self.bounds_range) / (2.0 * self.bounds_range)
        train_X_norm = torch.clamp(train_X_norm, 0.0, 1.0)

        train_Y = rewards.clone().detach().to(dtype=torch.double).unsqueeze(-1)

        # Track best reward for TuRBO adaptation
        current_max_y = float(train_Y.max())
        if current_max_y > self.best_reward + 1e-3:
            self.best_reward = current_max_y
            self.turbo_success_counter += 1
            self.turbo_failure_counter = 0
        else:
            self.turbo_failure_counter += 1
            self.turbo_success_counter = 0

        # TuRBO Length Update
        if self.turbo_success_counter >= self.turbo_success_tolerance:
            self.turbo_length = min(self.turbo_length * 2.0, self.turbo_length_max)
            self.turbo_success_counter = 0
        elif self.turbo_failure_counter >= self.turbo_failure_tolerance:
            self.turbo_length = self.turbo_length / 2.0
            self.turbo_failure_counter = 0

        # Restart trust region if collapsed below minimum
        if self.turbo_length < self.turbo_length_min:
            self.turbo_length = 0.8

        # Standardize train_Y for GP stability
        y_mean = train_Y.mean()
        y_std = train_Y.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1.0, dtype=torch.double)
        norm_Y = (train_Y - y_mean) / y_std

        # 1. Fit SingleTaskGP on unit hypercube
        gp = SingleTaskGP(train_X_norm, norm_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass # Fallback to default hyperparams if convergence fails

        # 2. Select Acquisition Function
        if acq_type.lower() == "qlogei":
            best_f = norm_Y.max()
            acq_func = qLogExpectedImprovement(model=gp, best_f=best_f)
        else:
            acq_func = qUpperConfidenceBound(gp, beta=beta)

        # 3. Determine Search Bounds (TuRBO Local Box vs Full Unit Hypercube)
        device = train_X_norm.device
        if self.use_turbo and len(train_X_norm) > 0:
            best_idx = torch.argmax(train_Y).item()
            x_center = train_X_norm[best_idx]
            half_len = self.turbo_length / 2.0
            lower = torch.clamp(x_center - half_len, 0.0, 1.0)
            upper = torch.clamp(x_center + half_len, 0.0, 1.0)
            # Ensure valid non-degenerate box
            upper = torch.max(upper, lower + 1e-4)
            bounds = torch.stack([lower, upper]).to(device)
        else:
            bounds = self.unit_bounds.to(device)

        # 4. Optimize Acquisition Function
        candidates_norm, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=q,
            num_restarts=4,
            raw_samples=16
        )

        # Unscale candidates back to [-bounds_range, bounds_range]
        candidates = candidates_norm * (2.0 * self.bounds_range) - self.bounds_range
        return candidates.float()
