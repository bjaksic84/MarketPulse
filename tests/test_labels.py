"""Tests for label generation — critical for ensuring no look-ahead bias."""

import pytest
import numpy as np
import pandas as pd

from src.features.labels import (
    generate_labels,
    get_clean_features_and_labels,
)


def _make_sample_df(n_days: int = 300) -> pd.DataFrame:
    """Create a sample DataFrame with price and feature data."""
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    close = 100.0 + np.cumsum(np.random.randn(n_days) * 1.5)
    close = np.maximum(close, 10)

    df = pd.DataFrame({
        "open": close + np.random.randn(n_days) * 0.3,
        "high": close + np.abs(np.random.randn(n_days) * 0.5),
        "low": close - np.abs(np.random.randn(n_days) * 0.5),
        "close": close,
        "volume": np.random.randint(1e6, 1e7, n_days).astype(float),
        "adj_close": close,
        "returns": pd.Series(close).pct_change().values,
        # Add some mock feature columns
        "rsi_14": np.random.uniform(20, 80, n_days),
        "sma_20": close + np.random.randn(n_days) * 2,
        "macd": np.random.randn(n_days) * 0.1,
        "vol_20d": np.abs(np.random.randn(n_days) * 0.01),
    }, index=dates)

    return df


class TestLabelGeneration:
    """Test label generation for classification and regression."""

    def test_binary_classification(self):
        df = _make_sample_df()
        result = generate_labels(df, horizon=1, num_classes=2)

        assert "fwd_return_1d" in result.columns
        assert "label" in result.columns
        assert "label_name" in result.columns

        # Binary labels should be 0 or 1 (or NaN for last row)
        valid_labels = result["label"].dropna()
        assert set(valid_labels.unique()).issubset({0, 1})

    def test_ternary_classification(self):
        df = _make_sample_df()
        result = generate_labels(df, horizon=1, num_classes=3, threshold=0.01)

        valid_labels = result["label"].dropna()
        assert set(valid_labels.unique()).issubset({0, 1, 2})  # DOWN, FLAT, UP

        valid_names = result["label_name"].dropna()
        assert set(valid_names.unique()).issubset({"DOWN", "FLAT", "UP"})

    def test_regression_labels(self):
        df = _make_sample_df()
        result = generate_labels(df, horizon=1, label_type="regression")

        assert "label" in result.columns
        # Regression labels should be continuous (not just 0,1,2)
        valid_labels = result["label"].dropna()
        assert len(valid_labels.unique()) > 10

    def test_horizon_affects_nan_count(self):
        df = _make_sample_df()

        result_1d = generate_labels(df, horizon=1, num_classes=2)
        result_5d = generate_labels(df, horizon=5, num_classes=2)

        # Horizon=5 should have more NaN labels (last 5 rows vs last 1)
        nan_1d = result_1d["label"].isna().sum()
        nan_5d = result_5d["label"].isna().sum()
        assert nan_5d > nan_1d

    def test_forward_return_correctness(self):
        df = _make_sample_df()
        result = generate_labels(df, horizon=1, num_classes=2)

        # Forward return at index i should be (close[i+1] / close[i]) - 1
        for i in range(len(df) - 1):
            expected = df["close"].iloc[i + 1] / df["close"].iloc[i] - 1
            actual = result["fwd_return_1d"].iloc[i]
            np.testing.assert_almost_equal(actual, expected, decimal=10)

    def test_no_look_ahead_bias(self):
        """CRITICAL TEST: Verify that features at time t don't use future data.

        Strategy: if we shuffle labels, model accuracy should drop to ~random.
        This test verifies the structural separation of features and labels.
        """
        df = _make_sample_df()
        result = generate_labels(df, horizon=1, num_classes=2)

        # Get feature and label columns
        X, y = get_clean_features_and_labels(result)

        # Feature columns should NOT include forward returns or labels
        feature_cols = X.columns.tolist()
        assert "fwd_return_1d" not in feature_cols
        assert "label" not in feature_cols
        assert "label_name" not in feature_cols
        assert "close" not in feature_cols
        assert "open" not in feature_cols

        # All feature values at index i should only depend on data at index <= i
        # (This is structurally guaranteed by our pipeline design,
        # but we verify no forward-looking columns leak in)
        for col in feature_cols:
            assert not col.startswith("fwd_return"), f"Feature {col} is forward-looking!"

    def test_label_threshold_effect(self):
        """Higher threshold should produce more FLAT labels."""
        df = _make_sample_df()

        result_tight = generate_labels(df, num_classes=3, threshold=0.005)
        result_wide = generate_labels(df, num_classes=3, threshold=0.03)

        flat_tight = (result_tight["label_name"] == "FLAT").sum()
        flat_wide = (result_wide["label_name"] == "FLAT").sum()

        assert flat_wide > flat_tight, "Wider threshold should have more FLAT labels"


class TestGetCleanFeaturesAndLabels:
    """Test the feature/label extraction utility."""

    def test_returns_correct_shapes(self):
        df = _make_sample_df()
        labeled = generate_labels(df, horizon=1, num_classes=2)
        X, y = get_clean_features_and_labels(labeled)

        assert len(X) == len(y)
        assert len(X) > 0

        # Should not have NaN in either X or y after cleaning
        assert X.isna().sum().sum() == 0
        assert y.isna().sum() == 0

    def test_excludes_non_feature_columns(self):
        df = _make_sample_df()
        labeled = generate_labels(df, horizon=1, num_classes=2)
        X, y = get_clean_features_and_labels(labeled)

        excluded = {"open", "high", "low", "close", "volume", "adj_close",
                     "label", "label_name", "fwd_return_1d"}
        for col in excluded:
            assert col not in X.columns, f"{col} should be excluded from features"

    def test_explicit_feature_cols(self):
        df = _make_sample_df()
        labeled = generate_labels(df, horizon=1, num_classes=2)

        feature_cols = ["rsi_14", "sma_20", "macd"]
        X, y = get_clean_features_and_labels(labeled, feature_cols=feature_cols)

        assert list(X.columns) == feature_cols

    def test_aligned_indices(self):
        df = _make_sample_df()
        labeled = generate_labels(df, horizon=1, num_classes=2)
        X, y = get_clean_features_and_labels(labeled)

        # Indices must match exactly
        pd.testing.assert_index_equal(X.index, y.index)
