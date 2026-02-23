"""
Label generation for supervised learning.

Generates classification and regression targets from price data.
CRITICAL: All labels are forward-looking (they use future prices).
Features must NEVER include label columns.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_labels(
    df: pd.DataFrame,
    horizon: int = 1,
    label_type: str = "classification",
    num_classes: int = 3,
    threshold: Union[float, str] = 0.01,
) -> pd.DataFrame:
    """Generate prediction labels (targets) from price data.

    Parameters
    ----------
    df : pd.DataFrame
        Must have column 'close'.
    horizon : int
        Number of trading days to look forward (default: 1 = next day).
    label_type : str
        'classification' or 'regression'.
    num_classes : int
        For classification:
        - 2: UP (>=0%) / DOWN (<0%)
        - 3: UP (>threshold) / FLAT / DOWN (<-threshold)
    threshold : float or str
        For 3-class classification:
        - float: Fixed threshold for the FLAT zone (e.g. 0.01 = ±1%).
        - "adaptive": Compute per-ticker thresholds from percentiles
          of the actual return distribution (p33/p67) to guarantee
          balanced classes (~33% each).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new label columns:
        - 'fwd_return_{horizon}d': raw forward return (always added)
        - 'label': classification label (0, 1, or 2) or regression target
        - 'label_name': human-readable label ('UP', 'DOWN', 'FLAT')
    """
    if df.empty:
        return df

    df = df.copy()

    # ── Forward Return (the raw prediction target) ──
    # This is the % return from close at time t to close at time t+horizon
    fwd_col = f"fwd_return_{horizon}d"
    df[fwd_col] = df["close"].shift(-horizon) / df["close"] - 1

    if label_type == "classification":
        df = _generate_classification_labels(
            df, fwd_col, num_classes, threshold
        )
    elif label_type == "regression":
        df = _generate_regression_labels(df, fwd_col)
    else:
        raise ValueError(f"Unknown label_type: '{label_type}'")

    # Log label distribution
    if "label" in df.columns:
        valid = df["label"].dropna()
        if len(valid) > 0:
            if label_type == "classification":
                dist = df["label_name"].value_counts(normalize=True)
                thresh_str = (
                    "adaptive" if isinstance(threshold, str)
                    else f"{threshold:.1%}"
                )
                logger.info(
                    f"Label distribution (horizon={horizon}d, "
                    f"threshold={thresh_str}):\n{dist.to_string()}"
                )
            else:
                logger.info(
                    f"Regression target stats (horizon={horizon}d):\n"
                    f"  mean={valid.mean():.4f}, std={valid.std():.4f}, "
                    f"  min={valid.min():.4f}, max={valid.max():.4f}"
                )

    return df


def _generate_classification_labels(
    df: pd.DataFrame,
    fwd_col: str,
    num_classes: int,
    threshold: Union[float, str],
) -> pd.DataFrame:
    """Generate classification labels from forward returns."""

    if num_classes == 2:
        # Binary: UP (return >= 0) = 1, DOWN (return < 0) = 0
        df["label"] = np.where(df[fwd_col] >= 0, 1, 0)
        df["label_name"] = np.where(df[fwd_col] >= 0, "UP", "DOWN")

    elif num_classes == 3:
        if isinstance(threshold, str) and threshold.lower() == "adaptive":
            # Adaptive: use percentile-based thresholds for balanced classes.
            # IMPORTANT: We compute thresholds from only the first 70% of
            # data to avoid look-ahead bias (the test set's return
            # distribution shouldn't influence the threshold definition).
            fwd_returns = df[fwd_col].dropna()
            n_threshold_window = int(len(fwd_returns) * 0.70)
            threshold_data = fwd_returns.iloc[:n_threshold_window]
            p33 = threshold_data.quantile(0.333)
            p67 = threshold_data.quantile(0.667)

            logger.info(
                f"Adaptive thresholds: p33={p33:.4f} ({p33:.2%}), "
                f"p67={p67:.4f} ({p67:.2%})"
            )

            conditions = [
                df[fwd_col] > p67,
                df[fwd_col] < p33,
            ]
            choices_num = [2, 0]      # 0=DOWN, 1=FLAT, 2=UP
            choices_name = ["UP", "DOWN"]
        else:
            # Fixed symmetric threshold: ±threshold
            threshold = float(threshold)
            conditions = [
                df[fwd_col] > threshold,
                df[fwd_col] < -threshold,
            ]
            choices_num = [2, 0]      # 0=DOWN, 1=FLAT, 2=UP
            choices_name = ["UP", "DOWN"]

        df["label"] = np.select(conditions, choices_num, default=1)
        df["label_name"] = np.select(conditions, choices_name, default="FLAT")

    else:
        raise ValueError(f"num_classes must be 2 or 3, got {num_classes}")

    # Set label to NaN where forward return is NaN (last `horizon` rows)
    mask = df[fwd_col].isna()
    df.loc[mask, "label"] = np.nan
    df.loc[mask, "label_name"] = np.nan

    return df


def _generate_regression_labels(
    df: pd.DataFrame,
    fwd_col: str,
) -> pd.DataFrame:
    """Generate regression labels (raw forward return)."""
    df["label"] = df[fwd_col]
    df["label_name"] = "regression"
    return df


def get_clean_features_and_labels(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    label_col: str = "label",
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and label vector y from a labeled DataFrame.

    Ensures no look-ahead bias by:
    1. Separating features from labels
    2. Dropping rows with NaN in either features or labels
    3. Never including forward-return or label columns in features

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with both feature and label columns.
    feature_cols : list, optional
        Explicit list of feature column names. If None, automatically
        excludes known non-feature columns.
    label_col : str
        Name of the label column (default: 'label').
    dropna : bool
        Whether to drop rows with any NaN values.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) — feature matrix and label vector with aligned indices.
    """
    # Columns that are NEVER features
    # 1. Target / label columns
    # 2. Raw OHLCV (price-level, non-stationary)
    # 3. Non-stationary derived columns (raw MAs, BB bands, ATR, OBV)
    #    These scale with price level and cause severe overfitting
    #    in walk-forward validation. Only their NORMALIZED versions
    #    (dist_sma_*, bb_width, bb_pct, atr_pct, obv_roc_*) are kept.
    exclude_cols = {
        # Targets
        "label", "label_name", "regime_label",
        # Raw OHLCV
        "open", "high", "low", "close", "volume", "adj_close",
        # Non-stationary price-level features
        "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26",
        "bb_lower", "bb_mid", "bb_upper",
        "atr_14",
        "obv",
        # Raw MACD values (keep normalized macd_pct, macd_hist_pct instead)
        "macd", "macd_hist", "macd_signal",
    }
    # Also exclude any forward return columns
    exclude_cols.update(
        col for col in df.columns if col.startswith("fwd_return")
    )

    if feature_cols is None:
        feature_cols = [
            col for col in df.columns if col not in exclude_cols
        ]

    X = df[feature_cols].copy()
    y = df[label_col].copy()

    if dropna:
        # Drop rows where EITHER features or labels have NaN
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

    logger.info(
        f"Clean dataset: {len(X)} samples, {len(feature_cols)} features"
    )

    return X, y
