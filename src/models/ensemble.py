"""
Ensemble model for MarketPulse.

Combines predictions from multiple classifiers (XGBoost + LightGBM)
using weighted soft-voting. The ensemble leverages complementary
strengths of different algorithms:

- XGBoost: depth-wise growth, strong regularization
- LightGBM: leaf-wise growth, histogram binning, faster

Ensembles typically outperform any single model by ~2-5% in F1 because
they reduce model-specific overfitting through prediction averaging.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .lightgbm_classifier import MarketPulseLGBClassifier
from .xgboost_classifier import MarketPulseXGBClassifier
from .xgboost_regressor import MarketPulseXGBRegressor
from .lightgbm_regressor import MarketPulseLGBRegressor

logger = logging.getLogger(__name__)


class MarketPulseEnsemble:
    """Weighted soft-voting ensemble of multiple classifiers.

    Combines class probability vectors from each model:

        P_ensemble(class c) = sum( w_i * P_i(class c) ) / sum(w_i)

    Parameters
    ----------
    num_classes : int
        Number of output classes (2 or 3).
    models_config : list of dict, optional
        Configuration for each model in the ensemble.
        Each dict has 'type' and 'weight' keys.
        If None, defaults to equal-weight XGBoost + LightGBM.
    strategy_config : dict, optional
        Full strategy config for hyperparameter extraction.
    """

    MODEL_REGISTRY = {
        "xgboost_classifier": MarketPulseXGBClassifier,
        "lightgbm_classifier": MarketPulseLGBClassifier,
        "xgboost_regressor": MarketPulseXGBRegressor,
        "lightgbm_regressor": MarketPulseLGBRegressor,
    }

    def __init__(
        self,
        num_classes: int = 3,
        models_config: Optional[List[Dict]] = None,
        strategy_config: Optional[Dict] = None,
    ):
        self.num_classes = num_classes
        self.strategy_config = strategy_config or {}

        # Default: equal-weight XGB + LGB
        if models_config is None:
            models_config = [
                {"type": "xgboost_classifier", "weight": 0.5},
                {"type": "lightgbm_classifier", "weight": 0.5},
            ]

        self.model_specs = models_config
        self.models: List[Tuple[str, float, object]] = []  # (name, weight, model)
        self.feature_names: list = []

        # Detect regression mode from model types
        self._is_regression = any(
            "regressor" in m.get("type", "") for m in models_config
        )

        # Normalize weights
        total_w = sum(m.get("weight", 1.0) for m in models_config)
        self._weights_normalized = [
            m.get("weight", 1.0) / total_w for m in models_config
        ]

    def _create_model(self, model_type: str):
        """Instantiate a model from the registry."""
        cls = self.MODEL_REGISTRY.get(model_type)
        if cls is None:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available: {list(self.MODEL_REGISTRY.keys())}"
            )
        return cls.from_strategy_config(self.strategy_config)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        balance_classes: bool = True,
    ) -> "MarketPulseEnsemble":
        """Train all models in the ensemble.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val : optional validation data (for early stopping)
        balance_classes : bool
            Pass to each sub-model.

        Returns
        -------
        self
        """
        self.feature_names = list(X_train.columns)
        self.models = []

        for i, spec in enumerate(self.model_specs):
            model_type = spec["type"]
            weight = self._weights_normalized[i]

            logger.info(f"Training ensemble member {i+1}/{len(self.model_specs)}: {model_type} (weight={weight:.2f})")

            model = self._create_model(model_type)
            model.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                balance_classes=balance_classes,
            )

            self.models.append((model_type, weight, model))

        logger.info(
            f"Ensemble trained: {len(self.models)} models, "
            f"{len(X_train)} samples, {len(self.feature_names)} features"
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of class probabilities.

        P_ensemble(c) = Σ w_i · P_i(c)

        For regression ensembles, returns predictions as (n_samples, 1).

        Returns
        -------
        np.ndarray
            Shape (n_samples, num_classes) for classification,
            or (n_samples, 1) for regression.
        """
        if not self.models:
            raise RuntimeError("Ensemble not trained. Call fit() first.")

        if self._is_regression:
            preds = self.predict(X)
            return preds.reshape(-1, 1)

        proba_sum = np.zeros((len(X), self.num_classes))

        for name, weight, model in self.models:
            proba = model.predict_proba(X)
            proba_sum += weight * proba

        return proba_sum

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels (classification) or values (regression)."""
        if self._is_regression:
            # Weighted-average of raw predictions
            preds_sum = np.zeros(len(X))
            for name, weight, model in self.models:
                preds_sum += weight * model.predict(X)
            return preds_sum

        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def get_feature_importance(self) -> pd.Series:
        """Weighted average feature importance across models.

        Returns
        -------
        pd.Series
            Feature importance indexed by name, sorted descending.
        """
        if not self.models:
            raise RuntimeError("Ensemble not trained.")

        combined = pd.Series(0.0, index=self.feature_names)

        for name, weight, model in self.models:
            try:
                imp = model.get_feature_importance()
                # Normalize to sum to 1
                imp_norm = imp / imp.sum() if imp.sum() > 0 else imp
                combined = combined.add(imp_norm * weight, fill_value=0)
            except Exception as e:
                logger.warning(f"Could not get importance from {name}: {e}")

        return combined.sort_values(ascending=False)

    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get per-model predictions for agreement analysis.

        Returns
        -------
        dict
            Model name → predicted labels array.
        """
        predictions = {}
        for name, weight, model in self.models:
            predictions[f"{name}_w{weight:.2f}"] = model.predict(X)
        return predictions

    def get_agreement_score(self, X: pd.DataFrame) -> np.ndarray:
        """Compute prediction agreement (0-1) across ensemble members.

        1.0 = all models agree, 0.0 = maximum disagreement.

        Returns
        -------
        np.ndarray
            Per-sample agreement score.
        """
        preds = self.get_individual_predictions(X)
        pred_matrix = np.column_stack(list(preds.values()))

        # For each sample, compute the fraction of models agreeing with the majority
        agreement = np.zeros(len(X))
        for i in range(len(X)):
            vals, counts = np.unique(pred_matrix[i], return_counts=True)
            agreement[i] = counts.max() / len(self.models)

        return agreement

    @classmethod
    def from_strategy_config(cls, strategy_config: dict) -> "MarketPulseEnsemble":
        """Create ensemble from strategy config.

        Uses the 'ensemble' section if present, otherwise falls back
        to a single-model setup using the 'model' section.
        """
        num_classes = strategy_config.get("num_classes", 3)
        ensemble_cfg = strategy_config.get("ensemble", {})

        if ensemble_cfg.get("enabled", False):
            models_config = ensemble_cfg.get("models", [])
            if not models_config:
                models_config = None
        else:
            # No ensemble configured — wrap the single model type
            model_type = strategy_config.get("model", {}).get("type", "xgboost_classifier")
            models_config = [{"type": model_type, "weight": 1.0}]

        return cls(
            num_classes=num_classes,
            models_config=models_config,
            strategy_config=strategy_config,
        )
