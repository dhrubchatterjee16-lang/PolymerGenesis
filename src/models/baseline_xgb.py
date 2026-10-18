"""
Baseline Model: XGBoost Regressor with Morgan Fingerprints and Physicochemical Descriptors.
Provides tree-variance based uncertainty estimation.
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from ..features.featurizer import get_morgan_fingerprint, get_rdkit_descriptors

class PolymerXGBoostBaseline:
    def __init__(self, n_estimators: int = 300, max_depth: int = 6, learning_rate: float = 0.05):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False

    def featurize_smiles_list(self, smiles_list: list) -> np.ndarray:
        """Constructs a concatenated feature matrix: [MorganFP (2048) | RDKit Descriptors (12)]."""
        features = []
        for s in smiles_list:
            fp = get_morgan_fingerprint(s)
            desc = get_rdkit_descriptors(s)
            features.append(np.concatenate([fp, desc]))
        return np.array(features, dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the XGBoost regressor."""
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray):
        """Predicts point estimates."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X)

    def predict_with_uncertainty(self, X: np.ndarray):
        """
        Estimates predictive uncertainty using variance across tree iterations.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        
        preds_mean = self.model.predict(X)
        booster = self.model.get_booster()
        dmatrix = xgb.DMatrix(X)
        
        # Iteration-level predictions
        num_trees = booster.num_boosted_rounds()
        # Collect outputs at multiple checkpoints across trees
        tree_preds = []
        step = max(1, num_trees // 15)
        for i in range(step, num_trees + 1, step):
            sub_pred = booster.predict(dmatrix, iteration_range=(0, i))
            tree_preds.append(sub_pred)
            
        tree_preds = np.array(tree_preds) # shape: (K, N)
        std_estimates = np.std(tree_preds, axis=0) # shape: (N,)
        
        return preds_mean, std_estimates
