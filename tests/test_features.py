"""Tests for feature engineering modules."""

import pytest
import numpy as np
import pandas as pd

from src.features.technical import compute_technical_indicators, get_feature_names
from src.features.returns import compute_return_features, get_return_feature_names


def _make_sample_ohlcv(n_days: int = 300) -> pd.DataFrame:
    """Create a realistic synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    # Random walk for price
    close = 100.0 + np.cumsum(np.random.randn(n_days) * 1.5)
    close = np.maximum(close, 10)  # prevent negative prices

    high = close + np.abs(np.random.randn(n_days) * 0.5)
    low = close - np.abs(np.random.randn(n_days) * 0.5)
    open_ = close + np.random.randn(n_days) * 0.3

    volume = np.random.randint(1_000_000, 10_000_000, size=n_days).astype(float)

    # Returns
    returns = pd.Series(close).pct_change().values
    log_returns = np.log(pd.Series(close) / pd.Series(close).shift(1)).values

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "adj_close": close,
        "returns": returns,
        "log_returns": log_returns,
    }, index=dates)

    return df


class TestTechnicalIndicators:
    """Test technical indicator computation."""

    def test_compute_all_indicators(self):
        df = _make_sample_ohlcv(300)
        result = compute_technical_indicators(df)

        # Should have new columns
        assert len(result.columns) > len(df.columns)

        # Check key indicators exist
        assert "sma_20" in result.columns
        assert "sma_50" in result.columns
        assert "rsi_14" in result.columns
        assert "macd" in result.columns
        assert "bb_upper" in result.columns
        assert "atr_14" in result.columns

    def test_rsi_range(self):
        df = _make_sample_ohlcv(300)
        result = compute_technical_indicators(df, include_groups=["momentum"])
        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= 0, "RSI should be >= 0"
        assert rsi.max() <= 100, "RSI should be <= 100"

    def test_sma_correctness(self):
        df = _make_sample_ohlcv(300)
        result = compute_technical_indicators(df, include_groups=["trend"])
        # SMA 20 should equal the rolling mean of last 20 close prices
        expected = df["close"].rolling(20).mean()
        pd.testing.assert_series_equal(
            result["sma_20"].dropna(),
            expected.dropna(),
            check_names=False,
            atol=0.01,
        )

    def test_bollinger_bands_order(self):
        df = _make_sample_ohlcv(300)
        result = compute_technical_indicators(df, include_groups=["volatility"])
        valid = result.dropna(subset=["bb_lower", "bb_mid", "bb_upper"])
        assert (valid["bb_lower"] <= valid["bb_mid"]).all()
        assert (valid["bb_mid"] <= valid["bb_upper"]).all()

    def test_custom_features(self):
        df = _make_sample_ohlcv(300)
        result = compute_technical_indicators(df)
        assert "sma_cross_20_50" in result.columns
        assert "dist_sma_20_pct" in result.columns
        assert "rsi_oversold" in result.columns
        assert "close_position" in result.columns

        # Binary features should be 0 or 1
        assert set(result["sma_cross_20_50"].dropna().unique()).issubset({0, 1})
        assert set(result["rsi_oversold"].dropna().unique()).issubset({0, 1})

    def test_selective_groups(self):
        df = _make_sample_ohlcv(300)
        result = compute_technical_indicators(df, include_groups=["momentum"])
        assert "rsi_14" in result.columns
        assert "sma_20" not in result.columns  # trend not included

    def test_get_feature_names(self):
        names = get_feature_names()
        assert len(names) > 20
        assert "rsi_14" in names
        assert "macd" in names

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = compute_technical_indicators(df)
        assert result.empty


class TestReturnFeatures:
    """Test return-based feature computation."""

    def test_compute_all_return_features(self):
        df = _make_sample_ohlcv(300)
        result = compute_return_features(df)

        assert "ret_1d" in result.columns
        assert "ret_5d" in result.columns
        assert "vol_5d" in result.columns
        assert "vol_20d" in result.columns
        assert "sharpe_20d" in result.columns
        assert "max_dd_20d" in result.columns
        assert "momentum_10d" in result.columns
        assert "zscore_20d" in result.columns
        assert "skew_20d" in result.columns
        assert "consec_up" in result.columns

    def test_lagged_returns_correctness(self):
        df = _make_sample_ohlcv(300)
        result = compute_return_features(df)

        # 5-day return should match manual calculation
        expected_5d = df["close"].pct_change(5)
        pd.testing.assert_series_equal(
            result["ret_5d"].dropna(),
            expected_5d.dropna(),
            check_names=False,
            atol=1e-10,
        )

    def test_volatility_positive(self):
        df = _make_sample_ohlcv(300)
        result = compute_return_features(df)
        vol = result["vol_20d"].dropna()
        assert (vol >= 0).all(), "Volatility should be non-negative"

    def test_max_drawdown_negative(self):
        df = _make_sample_ohlcv(300)
        result = compute_return_features(df)
        dd = result["max_dd_20d"].dropna()
        assert (dd <= 0).all(), "Max drawdown should be <= 0"

    def test_consecutive_counts_non_negative(self):
        df = _make_sample_ohlcv(300)
        result = compute_return_features(df)
        assert (result["consec_up"].dropna() >= 0).all()
        assert (result["consec_down"].dropna() >= 0).all()

    def test_get_return_feature_names(self):
        names = get_return_feature_names()
        assert len(names) > 15
        assert "ret_1d" in names
        assert "sharpe_20d" in names

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = compute_return_features(df)
        assert result.empty
