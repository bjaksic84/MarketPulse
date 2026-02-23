"""
LightGBM regressor wrapper for MarketPulse Phase 4.

Mirrors the LitGBM classifier API but for continuous return prediction.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)


class MarketPulseLGBRegressor:
    """LightGBM regressor for return magnitude prediction."""

    DEFAULT_PARAMS = {
        "max_depth": 5,
        "n_estimators": 500,
        "learning_rate": 0.02,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "min_child_weight": 5,
        "num_leaves": 31,
        "reg_alpha": 0.3,
        "reg_lambda": 2.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "objective": "regression",
    }

    def __init__(self, hyperparameters: Optional[Dict] = None):
        self.hyperparameters = {**self.DEFAULT_PARAMS}
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)

        self.model: Optional[LGBMRegressor] = None
        self.feature_names: list = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 30,
        **kwargs,
    ) -> "MarketPulseLGBRegressor":
        """Train the LightGBM regressor."""
        self.feature_names = list(X_train.columns)

        params = {**self.hyperparameters}
        self.model = LGBMRegressor(**params)

        fit_params = {"X": X_train, "y": y_train}

        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["callbacks"] = [
                __import__("lightgbm").early_stopping(early_stopping_rounds, verbose=False),
            ]

        self.model.fit(**fit_params)

        logger.info(
            f"Trained LGBMRegressor: {len(X_train)} samples, "
            f"{len(self.feature_names)} features"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict return magnitude."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """API compatibility — returns predictions as (n, 1)."""
        return self.predict(X).reshape(-1, 1)

    def get_feature_importance(self) -> pd.Series:
        """Feature importance (gain-based)."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

    @classmethod
    def from_strategy_config(cls, strategy_config: dict) -> "MarketPulseLGBRegressor":
        """Create from strategy config dict."""
        model_cfg = strategy_config.get("model", {})
        hyperparams = model_cfg.get("hyperparameters", {})
        return cls(hyperparameters=hyperparams)
