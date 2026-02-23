"""
LightGBM classifier wrapper for MarketPulse.

LightGBM offers complementary strengths to XGBoost:
- Histogram-based splitting → faster training on large datasets
- Leaf-wise growth → captures complex patterns
- Native categorical support
- Lower memory usage

Used as the second model in the Phase 3 ensemble.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)


class MarketPulseLGBClassifier:
    """LightGBM-based price movement classifier.

    Parameters
    ----------
    hyperparameters : dict, optional
        LightGBM hyperparameters. Merged with sensible defaults.
    num_classes : int
        Number of output classes (2 or 3).
    """

    DEFAULT_PARAMS = {
        "max_depth": 4,
        "n_estimators": 200,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 10,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "num_leaves": 15,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    def __init__(
        self,
        hyperparameters: Optional[Dict] = None,
        num_classes: int = 3,
        calibrate_threshold: bool = True,
    ):
        self.num_classes = num_classes
        self.calibrate_threshold = calibrate_threshold
        self.optimal_threshold = 0.5
        self.hyperparameters = {**self.DEFAULT_PARAMS}
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)

        # Set objective
        if num_classes == 2:
            self.hyperparameters["objective"] = "binary"
            self.hyperparameters["metric"] = "binary_logloss"
        else:
            self.hyperparameters["objective"] = "multiclass"
            self.hyperparameters["metric"] = "multi_logloss"
            self.hyperparameters["num_class"] = num_classes

        self.model = None
        self.feature_names: list = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        balance_classes: bool = True,
        early_stopping_rounds: int = 30,
    ) -> "MarketPulseLGBClassifier":
        """Train the LightGBM classifier."""
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        self.feature_names = list(X_train.columns)
        y_train = y_train.astype(int)

        params = {**self.hyperparameters}
        # Remove keys not accepted by LGBMClassifier
        params.pop("metric", None)

        self.model = LGBMClassifier(**params)

        # Sample weights for class balancing
        sample_weight = None
        if balance_classes:
            sample_weight = compute_sample_weight("balanced", y_train)

        fit_params = {
            "X": X_train,
            "y": y_train,
            "sample_weight": sample_weight,
        }

        callbacks = [log_evaluation(period=0)]  # Suppress output

        if X_val is not None and y_val is not None:
            callbacks.append(early_stopping(early_stopping_rounds, verbose=False))
            fit_params["eval_set"] = [(X_val, y_val.astype(int))]
            fit_params["callbacks"] = callbacks
        else:
            fit_params["callbacks"] = callbacks

        self.model.fit(**fit_params)

        # Calibrate prediction threshold
        if self.calibrate_threshold and self.num_classes == 2:
            self._calibrate_threshold(X_train, y_train)

        logger.info(
            f"Trained LightGBM: {len(X_train)} samples, "
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
        """Find the optimal probability threshold on a held-out validation split."""
        n = len(X_train)
        val_size = max(min_val_size, int(n * val_fraction))
        if val_size >= n - 50:
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
        determined by comparing P(class=1) against self.optimal_threshold.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.num_classes == 2 and self.optimal_threshold != 0.5:
            proba = self.model.predict_proba(X)[:, 1]
            return (proba >= self.optimal_threshold).astype(int)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores (gain-based)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        importance = self.model.feature_importances_
        return (
            pd.Series(importance, index=self.feature_names)
            .sort_values(ascending=False)
        )

    @classmethod
    def from_strategy_config(
        cls, strategy_config: dict
    ) -> "MarketPulseLGBClassifier":
        """Create from a strategy config dict."""
        model_config = strategy_config.get("model", {})
        hyperparams = model_config.get("hyperparameters", {})
        num_classes = strategy_config.get("num_classes", 3)

        # Filter to LightGBM-compatible params
        lgb_params = {}
        valid_keys = set(cls.DEFAULT_PARAMS.keys()) | {
            "max_depth", "n_estimators", "learning_rate", "subsample",
            "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda",
            "num_leaves", "random_state", "n_jobs", "verbosity",
        }
        for k, v in hyperparams.items():
            if k in valid_keys:
                lgb_params[k] = v

        return cls(
            hyperparameters=lgb_params,
            num_classes=num_classes,
            calibrate_threshold=False,  # ensemble handles calibration
        )
