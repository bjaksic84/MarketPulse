"""
Feature selection for MarketPulse.

Provides multiple methods to reduce the 60-feature set to
the most predictive subset, reducing overfitting and noise.

Methods:
- Importance-based: keep top N by XGBoost gain
- Mutual information: information-theoretic relevance
- Correlation filter: remove redundant features (|r| > threshold)
- Combined pipeline: correlation filter → importance ranking
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger(__name__)


def select_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 15,
    model=None,
) -> List[str]:
    """Select top features by XGBoost gain importance.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Labels (integer-encoded).
    max_features : int
        Number of features to keep.
    model : optional
        Pre-trained model. If None, trains a quick XGBoost.

    Returns
    -------
    List[str]
        Selected feature names, sorted by importance descending.
    """
    if model is None:
        from src.models.xgboost_classifier import MarketPulseXGBClassifier
        model = MarketPulseXGBClassifier(
            hyperparameters={"n_estimators": 100, "max_depth": 4},
            num_classes=len(y.unique()),
        )
        model.fit(X, y, balance_classes=True)

    importance = model.get_feature_importance()
    selected = importance.head(max_features).index.tolist()

    logger.info(
        f"Importance selection: {len(selected)}/{len(X.columns)} features kept"
    )
    return selected


def select_by_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 15,
    random_state: int = 42,
) -> List[str]:
    """Select top features by mutual information with the target.

    MI measures how much knowing a feature reduces uncertainty about the label.
    Works for both continuous and discrete features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (NaN-free).
    y : pd.Series
        Labels.
    max_features : int
        Number of features to keep.

    Returns
    -------
    List[str]
        Selected feature names, sorted by MI descending.
    """
    mi_scores = mutual_info_classif(
        X, y.astype(int), random_state=random_state, n_neighbors=5
    )
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    selected = mi_series.head(max_features).index.tolist()

    logger.info(
        f"Mutual info selection: {len(selected)}/{len(X.columns)} features kept"
    )
    logger.info(f"Top 5 MI scores: {mi_series.head(5).to_dict()}")

    return selected


def filter_correlated(
    X: pd.DataFrame,
    threshold: float = 0.90,
    importance_order: Optional[List[str]] = None,
) -> List[str]:
    """Remove highly correlated features, keeping the more important one.

    For each pair with |correlation| > threshold, the feature with lower
    importance (or later in column order) is dropped.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    threshold : float
        Max allowed absolute correlation (default 0.90).
    importance_order : list, optional
        Features sorted by importance (most important first).
        If provided, the more important feature in each correlated pair is kept.

    Returns
    -------
    List[str]
        Surviving feature names.
    """
    corr_matrix = X.corr().abs()

    # Upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    # If we have importance ordering, use it to decide which to drop
    if importance_order:
        rank = {f: i for i, f in enumerate(importance_order)}
    else:
        rank = {f: i for i, f in enumerate(X.columns)}

    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        for corr_feat in correlated:
            # Drop the less important one
            if rank.get(col, 999) < rank.get(corr_feat, 999):
                to_drop.add(corr_feat)
            else:
                to_drop.add(col)

    surviving = [c for c in X.columns if c not in to_drop]

    logger.info(
        f"Correlation filter (threshold={threshold}): "
        f"dropped {len(to_drop)}, kept {len(surviving)}/{len(X.columns)}"
    )
    if to_drop:
        logger.debug(f"Dropped features: {sorted(to_drop)}")

    return surviving


def select_features_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 15,
    corr_threshold: float = 0.90,
    method: str = "importance",
) -> Tuple[List[str], pd.DataFrame]:
    """Combined feature selection pipeline.

    Steps:
    1. Remove highly correlated features (|r| > corr_threshold)
    2. Rank remaining by chosen method (importance or mutual_info)
    3. Keep top max_features

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Labels.
    max_features : int
        Final number of features to keep.
    corr_threshold : float
        Correlation filter threshold.
    method : str
        Ranking method: 'importance' or 'mutual_info'.

    Returns
    -------
    Tuple[List[str], pd.DataFrame]
        (selected_feature_names, feature_scores_dataframe)
    """
    logger.info(
        f"Feature selection pipeline: method={method}, "
        f"max_features={max_features}, corr_threshold={corr_threshold}"
    )

    # Step 1: Quick importance ranking for correlation filter priority
    from src.models.xgboost_classifier import MarketPulseXGBClassifier
    quick_model = MarketPulseXGBClassifier(
        hyperparameters={"n_estimators": 100, "max_depth": 4},
        num_classes=len(y.unique()),
    )
    quick_model.fit(X, y, balance_classes=True)
    importance = quick_model.get_feature_importance()
    importance_order = importance.index.tolist()

    # Step 2: Correlation filter (drop redundant, keep important)
    surviving = filter_correlated(
        X, threshold=corr_threshold, importance_order=importance_order
    )
    X_filtered = X[surviving]

    # Step 3: Rank by chosen method
    if method == "importance":
        selected = select_by_importance(
            X_filtered, y, max_features=max_features
        )
    elif method == "mutual_info":
        selected = select_by_mutual_info(
            X_filtered, y, max_features=max_features
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build scores DataFrame for analysis
    scores = pd.DataFrame({"feature": X.columns})
    scores["xgb_importance"] = scores["feature"].map(
        importance.to_dict()
    ).fillna(0)

    mi_scores = mutual_info_classif(
        X, y.astype(int), random_state=42, n_neighbors=5
    )
    scores["mutual_info"] = mi_scores
    scores["selected"] = scores["feature"].isin(selected)
    scores["corr_survived"] = scores["feature"].isin(surviving)
    scores = scores.sort_values("xgb_importance", ascending=False).reset_index(drop=True)

    logger.info(f"Final selection: {selected}")

    return selected, scores
