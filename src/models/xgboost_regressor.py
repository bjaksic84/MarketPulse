"""
XGBoost regressor wrapper for MarketPulse Phase 4.

Predicts return magnitude (continuous value) instead of direction class.
Shares the same API pattern as MarketPulseXGBClassifier.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class MarketPulseXGBRegressor:
    """XGBoost regressor for return magnitude prediction.

    Parameters
    ----------
    hyperparameters : dict, optional
        XGBoost hyperparameters. Merged with defaults.
    """

    DEFAULT_PARAMS = {
        "max_depth": 5,
        "n_estimators": 500,
        "learning_rate": 0.02,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "min_child_weight": 5,
        "gamma": 0.15,
        "reg_alpha": 0.3,
        "reg_lambda": 2.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "objective": "reg:squarederror",
    }

    def __init__(self, hyperparameters: Optional[Dict] = None):
        self.hyperparameters = {**self.DEFAULT_PARAMS}
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)

        self.model: Optional[XGBRegressor] = None
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: list = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 30,
        **kwargs,
    ) -> "MarketPulseXGBRegressor":
        """Train the regressor."""
        self.feature_names = list(X_train.columns)

        params = {**self.hyperparameters}
        if X_val is not None and y_val is not None:
            params["early_stopping_rounds"] = early_stopping_rounds

        self.model = XGBRegressor(**params)

        fit_params = {"X": X_train, "y": y_train, "verbose": False}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]

        self.model.fit(**fit_params)

        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"SHAP explainer init failed: {e}")

        logger.info(
            f"Trained XGBRegressor: {len(X_train)} samples, "
            f"{len(self.feature_names)} features"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict return magnitude."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """For API compatibility — returns predictions as (n, 1) array."""
        return self.predict(X).reshape(-1, 1)

    def get_feature_importance(self) -> pd.Series:
        """Feature importance (gain-based)."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values for regression."""
        if self.shap_explainer is None:
            raise RuntimeError("SHAP not available.")
        return self.shap_explainer.shap_values(X)

    @classmethod
    def from_strategy_config(cls, strategy_config: dict) -> "MarketPulseXGBRegressor":
        """Create from strategy config dict."""
        model_cfg = strategy_config.get("model", {})
        hyperparams = model_cfg.get("hyperparameters", {})
        return cls(hyperparameters=hyperparams)
