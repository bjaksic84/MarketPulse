"""
XGBoost classifier wrapper for MarketPulse.

Wraps xgboost.XGBClassifier with:
- Configuration from strategy YAML
- SHAP integration for explainability
- Class weight handling for imbalanced labels
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class MarketPulseXGBClassifier:
    """XGBoost-based price movement classifier with SHAP explainability.

    Parameters
    ----------
    hyperparameters : dict, optional
        XGBoost hyperparameters. Merged with sensible defaults.
    num_classes : int
        Number of output classes (2 or 3).
    """

    DEFAULT_PARAMS = {
        "max_depth": 3,
        "n_estimators": 150,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 10,
        "gamma": 0.5,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    def __init__(
        self,
        hyperparameters: Optional[Dict] = None,
        num_classes: int = 3,
        calibrate_threshold: bool = True,
    ):
        self.num_classes = num_classes
        self.calibrate_threshold = calibrate_threshold
        self.optimal_threshold = 0.5  # default; updated during fit
        self.hyperparameters = {**self.DEFAULT_PARAMS}
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)

        # Set objective based on number of classes
        if num_classes == 2:
            self.hyperparameters["objective"] = "binary:logistic"
            self.hyperparameters["eval_metric"] = "logloss"
        else:
            self.hyperparameters["objective"] = "multi:softprob"
            self.hyperparameters["eval_metric"] = "mlogloss"
            self.hyperparameters["num_class"] = num_classes

        self.model: Optional[XGBClassifier] = None
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: list = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        balance_classes: bool = True,
        early_stopping_rounds: int = 30,
    ) -> "MarketPulseXGBClassifier":
        """Train the XGBoost classifier.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels (integer-encoded).
        X_val : pd.DataFrame, optional
            Validation features (for early stopping).
        y_val : pd.Series, optional
            Validation labels.
        balance_classes : bool
            Whether to compute sample weights for class balancing.
        early_stopping_rounds : int
            Stop training if validation metric doesn't improve.

        Returns
        -------
        self
        """
        self.feature_names = list(X_train.columns)

        # Ensure labels are integer
        y_train = y_train.astype(int)

        # Create model
        params = {**self.hyperparameters}
        if X_val is not None and y_val is not None:
            params["early_stopping_rounds"] = early_stopping_rounds

        self.model = XGBClassifier(**params)

        # Compute sample weights for class balancing
        sample_weight = None
        if balance_classes:
            sample_weight = compute_sample_weight("balanced", y_train)

        # Fit
        fit_params = {
            "X": X_train,
            "y": y_train,
            "sample_weight": sample_weight,
            "verbose": False,
        }

        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val.astype(int))]

        self.model.fit(**fit_params)

        # Initialize SHAP explainer
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"SHAP explainer initialization failed: {e}")

        # Calibrate prediction threshold using trailing validation split
        if self.calibrate_threshold and self.num_classes == 2:
            self._calibrate_threshold(X_train, y_train)

        logger.info(
            f"Trained XGBoost: {len(X_train)} samples, "
            f"{len(self.feature_names)} features, "
            f"{self.num_classes} classes"
            f"{f', threshold={self.optimal_threshold:.2f}' if self.num_classes == 2 else ''}"
        )

        return self

    def _calibrate_threshold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        val_fraction: float = 0.15,
        min_val_size: int = 42,
    ):
        """Find the optimal probability threshold on a held-out validation split.

        Instead of always predicting UP when P(UP) >= 0.5, we search for the
        threshold that maximizes accuracy on the last `val_fraction` of training
        data. This accounts for class imbalance and model calibration issues.
        """
        n = len(X_train)
        val_size = max(min_val_size, int(n * val_fraction))
        if val_size >= n - 50:
            # Not enough data to split — keep default
            return

        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:].astype(int)

        proba = self.model.predict_proba(X_val)[:, 1]

        best_t, best_acc = 0.5, 0.0
        for t in np.arange(0.30, 0.70, 0.02):
            preds = (proba >= t).astype(int)
            acc = (preds == y_val.values).mean()
            if acc > best_acc:
                best_acc = acc
                best_t = t

        self.optimal_threshold = best_t
        logger.debug(f"Calibrated threshold: {best_t:.2f} (val acc: {best_acc:.3f})")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        For binary classification with a calibrated threshold, the label is
        determined by comparing P(class=1) against self.optimal_threshold
        instead of using the default 0.5 cutoff.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.num_classes == 2 and self.optimal_threshold != 0.5:
            proba = self.model.predict_proba(X)[:, 1]
            return (proba >= self.optimal_threshold).astype(int)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, num_classes).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores (gain-based).

        Returns
        -------
        pd.Series
            Feature importance indexed by feature name, sorted descending.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        importance = self.model.feature_importances_
        return (
            pd.Series(importance, index=self.feature_names)
            .sort_values(ascending=False)
        )

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for the given samples.

        Returns
        -------
        np.ndarray
            SHAP values. Shape depends on num_classes:
            - Binary: (n_samples, n_features)
            - Multi-class: (n_samples, n_features, n_classes)
        """
        if self.shap_explainer is None:
            raise RuntimeError(
                "SHAP explainer not available. Train with fit() first."
            )
        return self.shap_explainer.shap_values(X)

    def get_shap_explanation(self, X: pd.DataFrame) -> shap.Explanation:
        """Get full SHAP Explanation object for visualization."""
        if self.shap_explainer is None:
            raise RuntimeError("SHAP explainer not available.")
        return self.shap_explainer(X)

    @classmethod
    def from_strategy_config(
        cls, strategy_config: dict
    ) -> "MarketPulseXGBClassifier":
        """Create from a strategy config dict."""
        model_config = strategy_config.get("model", {})
        hyperparams = model_config.get("hyperparameters", {})
        num_classes = strategy_config.get("num_classes", 3)
        return cls(
            hyperparameters=hyperparams,
            num_classes=num_classes,
            calibrate_threshold=False,  # ensemble handles calibration
        )
