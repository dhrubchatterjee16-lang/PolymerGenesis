"""
Evaluation metrics for regression and Uncertainty Quantification (UQ).
Includes:
- Standard regression: R2, RMSE, MAE
- UQ Calibration: Expected vs Observed confidence coverage
- UQ Sparsification: Monotonic error reduction by discarding high-uncertainty predictions
"""

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Cross-version trapezoid integration across NumPy 1.x and NumPy 2.x
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes R2, RMSE, and MAE."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae)
    }

def compute_calibration_curve(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray, num_bins: int = 20):
    """
    Computes the UQ calibration curve:
    For theoretical confidence levels p in [0.05, 0.95],
    evaluates whether |y_true - y_pred| <= z_p * y_pred_std matches p.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_mean = np.asarray(y_pred_mean).ravel()
    y_pred_std = np.asarray(y_pred_std).ravel()

    # Avoid zero std
    y_pred_std = np.clip(y_pred_std, 1e-6, None)

    confidence_levels = np.linspace(0.05, 0.95, num_bins)
    observed_frequencies = []

    for p in confidence_levels:
        # Two-tailed standard normal critical value
        z = stats.norm.ppf((1.0 + p) / 2.0)
        bound = z * y_pred_std
        in_interval = np.abs(y_true - y_pred_mean) <= bound
        observed_frequencies.append(np.mean(in_interval))

    # Calibration error: Mean Absolute Calibration Error (MACE)
    mace = np.mean(np.abs(np.array(observed_frequencies) - confidence_levels))

    return {
        "expected_confidence": confidence_levels.tolist(),
        "observed_confidence": observed_frequencies,
        "MACE": float(mace)
    }

def compute_sparsification_curve(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray, num_points: int = 50):
    """
    Evaluates how RMSE decreases as we prune the most uncertain predictions.
    If the model's uncertainty is meaningful, dropping high-uncertainty samples
    will monotonically decrease the test RMSE.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_mean = np.asarray(y_pred_mean).ravel()
    y_pred_std = np.asarray(y_pred_std).ravel()

    # Sort descending by uncertainty (standard deviation)
    sort_idx = np.argsort(y_pred_std)[::-1]
    sorted_true = y_true[sort_idx]
    sorted_pred = y_pred_mean[sort_idx]

    fractions_removed = np.linspace(0.0, 0.8, num_points)
    rmse_curve = []
    n_samples = len(y_true)

    for frac in fractions_removed:
        num_remove = int(frac * n_samples)
        keep_true = sorted_true[num_remove:]
        keep_pred = sorted_pred[num_remove:]
        if len(keep_true) > 0:
            err = np.sqrt(mean_squared_error(keep_true, keep_pred))
            rmse_curve.append(float(err))
        else:
            rmse_curve.append(0.0)

    ause = float(_trapezoid(rmse_curve, fractions_removed))
    return {
        "fractions_removed": fractions_removed.tolist(),
        "rmse_curve": rmse_curve,
        "AUSE": ause
    }
