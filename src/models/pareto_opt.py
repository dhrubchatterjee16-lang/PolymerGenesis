"""
Multi-Objective Pareto Optimization Engine for PolymerGenesis.
Implements:
1. BoTorch Multi-Output Gaussian Process (SingleTaskGP) on normalized unit hypercube.
2. Fast Non-Dominated Box Partitioning.
3. q-Expected Hypervolume Improvement (qEHVI) acquisition function.
4. Non-dominated Pareto frontier extraction across multi-property tradeoffs (Tg, Eg, Circularity, Synthesizability).
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.cpp_extension")
warnings.filterwarnings("ignore", category=UserWarning, module="botorch.acquisition.multi_objective.logei")

import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
try:
    from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement as qEHVIClass
except ImportError:
    from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement as qEHVIClass
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.optim import optimize_acqf



class MultiObjectiveParetoOptimizer:
    def __init__(self, latent_dim: int = 64, num_objectives: int = 2, bounds_range: float = 3.0):
        self.latent_dim = latent_dim
        self.num_objectives = num_objectives
        self.bounds_range = bounds_range
        # Standard unit hypercube bounds [0.0, 1.0]^D
        self.unit_bounds = torch.tensor([
            [0.0] * latent_dim,
            [1.0] * latent_dim
        ], dtype=torch.double)

    def _scale_X(self, train_X: torch.Tensor) -> torch.Tensor:
        """Min-max scales continuous latent coordinates to [0, 1]^D."""
        X_d = train_X.clone().detach().to(dtype=torch.double)
        X_norm = (X_d + self.bounds_range) / (2.0 * self.bounds_range)
        return torch.clamp(X_norm, 0.0, 1.0)

    def _unscale_X(self, candidates_norm: torch.Tensor) -> torch.Tensor:
        """Unscales unit hypercube coordinates back to [-bounds_range, bounds_range]."""
        return candidates_norm * (2.0 * self.bounds_range) - self.bounds_range

    def optimize_pareto_step(self, z_history: torch.Tensor, outcomes: torch.Tensor, q: int = 4, ref_point: list = None):
        """
        Fits a Multi-Output Gaussian Process and optimizes q-Expected Hypervolume Improvement (qEHVI)
        to identify candidate latent points along the non-dominated Pareto frontier.

        Args:
            z_history: (N, D) observed latent vectors
            outcomes: (N, M) observed objective values (all formulated as MAXIMIZATION)
            q: number of candidate points to propose simultaneously
            ref_point: reference point vector in objective space (length M)
        """
        train_X_norm = self._scale_X(z_history)
        train_Y_raw = outcomes.clone().detach().to(dtype=torch.double)

        # Standardize train_Y per objective dimension for numerical stability
        y_mean = train_Y_raw.mean(dim=0, keepdim=True)
        y_std = train_Y_raw.std(dim=0, keepdim=True)
        y_std = torch.where(y_std < 1e-6, torch.ones_like(y_std), y_std)
        train_Y_norm = (train_Y_raw - y_mean) / y_std

        # Default normalized reference point (set below minimum observed outcome)
        if ref_point is None:
            norm_ref_point = (train_Y_norm.min(dim=0).values - 0.5).tolist()
        else:
            ref_tensor = torch.tensor(ref_point, dtype=torch.double, device=train_Y_raw.device).unsqueeze(0)
            norm_ref_point = ((ref_tensor - y_mean) / y_std).squeeze(0).tolist()

        # 1. Fit Multi-Output GP
        gp = SingleTaskGP(train_X_norm, train_Y_norm)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass

        # 2. Fast Non-Dominated Box Partitioning
        partitioning = FastNondominatedPartitioning(
            ref_point=torch.tensor(norm_ref_point, dtype=torch.double, device=train_X_norm.device),
            Y=train_Y_norm
        )

        # 3. qEHVI Acquisition Function (log-scale numerically stable)
        qehvi = qEHVIClass(
            model=gp,
            ref_point=torch.tensor(norm_ref_point, dtype=torch.double, device=train_X_norm.device),
            partitioning=partitioning
        )


        # 4. Optimize Acquisition Function over unit hypercube
        candidates_norm, _ = optimize_acqf(
            acq_function=qehvi,
            bounds=self.unit_bounds.to(train_X_norm.device),
            q=q,
            num_restarts=4,
            raw_samples=16
        )

        # Unscale candidates back to continuous latent space
        candidates = self._unscale_X(candidates_norm)
        return candidates.float()

    @staticmethod
    def identify_pareto_front(outcomes: np.ndarray) -> np.ndarray:
        """
        Calculates the non-dominated Pareto frontier mask for multi-objective outcomes (maximization).
        Returns boolean array where True indicates a Pareto-optimal point.
        """
        n_samples = outcomes.shape[0]
        is_pareto = np.ones(n_samples, dtype=bool)

        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    # Point j dominates point i if all elements >= and at least one >
                    if np.all(outcomes[j] >= outcomes[i]) and np.any(outcomes[j] > outcomes[i]):
                        is_pareto[i] = False
                        break

        return is_pareto
