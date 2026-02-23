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
        "max_depth": 6,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "num_leaves": 31,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    def __init__(
        self,
        hyperparameters: Optional[Dict] = None,
        num_classes: int = 3,
    ):
        self.num_classes = num_classes
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

        logger.info(
            f"Trained LightGBM: {len(X_train)} samples, "
            f"{len(self.feature_names)} features, "
            f"{self.num_classes} classes"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
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

        return cls(hyperparameters=lgb_params, num_classes=num_classes)
