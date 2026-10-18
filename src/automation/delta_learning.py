"""
Closed-Loop Sim-to-Real Delta-Learning Calibrator for PolymerGenesis Self-Driving Lab.
Implements:
1. Multi-fidelity discrepancy modeling: y_exp(x) = y_sim(x) + delta(x) + epsilon.
2. Gaussian Process modeling on residual delta(x) using Matérn 5/2 kernel with homoscedastic noise.
3. Automated posterior uncertainty propagation: sigma_cal^2 = sigma_sim^2 + sigma_delta^2.
4. Continuous real-time updating from at-line GPC / DSC / tensile analytics.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class DeltaLearningCalibrator:
    """
    Sim-to-Real Transfer Calibrator using Gaussian Process Delta-Learning.
    Corrects systematic physics/surrogate simulation errors using physical robotic feedback.
    """
    def __init__(self, property_name: str = "glass_transition_temp_C"):
        self.property_name = property_name
        self.X_history: List[np.ndarray] = []
        self.y_sim_history: List[float] = []
        self.y_exp_history: List[float] = []
        self.metadata_history: List[dict] = []

        # Robust Matérn 5/2 GP kernel with noise floor
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-4, 1e2))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=42
        )
        self.is_fitted = False

    def add_experimental_point(
        self,
        features: np.ndarray,
        y_sim: float,
        y_exp: float,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Ingests a new robotic at-line measurement tuple and re-calibrates delta GP.
        """
        feat = np.asarray(features, dtype=float).ravel()
        self.X_history.append(feat)
        self.y_sim_history.append(float(y_sim))
        self.y_exp_history.append(float(y_exp))
        self.metadata_history.append(metadata or {})

        # Retrain GP on all historical residuals delta = y_exp - y_sim
        X_arr = np.array(self.X_history)
        y_residuals = np.array(self.y_exp_history) - np.array(self.y_sim_history)

        if len(self.X_history) >= 2:
            try:
                self.gp.fit(X_arr, y_residuals)
                self.is_fitted = True
            except Exception:
                self.is_fitted = False

    def predict_calibrated(
        self,
        features: np.ndarray,
        y_sim_pred: Union[float, np.ndarray],
        sim_uncertainty: Union[float, np.ndarray] = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Outputs calibrated prediction and combined uncertainty:
        y_cal = y_sim + delta_mean
        sigma_cal = sqrt(sigma_sim^2 + sigma_delta^2)
        """
        feat = np.asarray(features, dtype=float)
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)

        y_sim_arr = np.atleast_1d(np.asarray(y_sim_pred, dtype=float))
        sim_unc_arr = np.atleast_1d(np.asarray(sim_uncertainty, dtype=float))

        if not self.is_fitted or len(self.X_history) < 2:
            # Fall back to pure simulation if insufficient experimental anchors
            return y_sim_arr, sim_unc_arr

        delta_mean, delta_std = self.gp.predict(feat, return_std=True)

        y_cal = y_sim_arr + delta_mean
        combined_std = np.sqrt(np.maximum(1e-6, sim_unc_arr ** 2 + delta_std ** 2))

        return y_cal, combined_std

    def get_summary_metrics(self) -> Dict[str, float]:
        """
        Computes sim-to-real discrepancy metrics (Sim MAE vs. Delta MAE).
        """
        n_pts = len(self.X_history)
        if n_pts == 0:
            return {"num_anchors": 0, "sim_mae": 0.0, "delta_mae": 0.0, "error_reduction_pct": 0.0}

        y_sim = np.array(self.y_sim_history)
        y_exp = np.array(self.y_exp_history)
        sim_mae = float(np.mean(np.abs(y_exp - y_sim)))

        if not self.is_fitted or n_pts < 2:
            return {
                "num_anchors": n_pts,
                "sim_mae": sim_mae,
                "delta_mae": sim_mae,
                "error_reduction_pct": 0.0
            }

        # Leave-one-out or fitted residual error
        X_arr = np.array(self.X_history)
        delta_pred, _ = self.gp.predict(X_arr, return_std=True)
        y_cal = y_sim + delta_pred
        delta_mae = float(np.mean(np.abs(y_exp - y_cal)))
        reduction = max(0.0, (sim_mae - delta_mae) / (sim_mae + 1e-6) * 100.0)

        return {
            "num_anchors": n_pts,
            "sim_mae": sim_mae,
            "delta_mae": delta_mae,
            "error_reduction_pct": reduction
        }

    def get_history(self) -> List[dict]:
        """Returns structured history of all experimental calibration cycles."""
        records = []
        for i in range(len(self.X_history)):
            records.append({
                "cycle": i + 1,
                "y_sim": self.y_sim_history[i],
                "y_exp": self.y_exp_history[i],
                "residual_delta": self.y_exp_history[i] - self.y_sim_history[i],
                "metadata": self.metadata_history[i]
            })
        return records
