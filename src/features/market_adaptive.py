"""
Market-adaptive features for MarketPulse Phase 3.

Generates features tailored to each market's unique behavior:

- **Indices**: Mean-reversion, VIX-proxy, breadth approximation
- **Stocks**: Gap analysis, earnings-surprise proxy, sector correlation

These features complement the universal technical + return features
and give each market-specific model an edge.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────── Shared / Universal Adaptive Features ──────────────────


def compute_volatility_regime(
    df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 60,
) -> pd.DataFrame:
    """Detect volatility regime by comparing short vs long rolling vol.

    Features:
    - vol_regime_ratio: short_vol / long_vol (>1 = expanding, <1 = contracting)
    - vol_regime_label: 0=low, 1=normal, 2=high (tercile of ratio)
    - vol_expansion: bool, short_vol > long_vol
    """
    if "returns" not in df.columns:
        return df

    short_vol = df["returns"].rolling(short_window).std()
    long_vol = df["returns"].rolling(long_window).std()

    ratio = short_vol / long_vol.replace(0, np.nan)
    df["vol_regime_ratio"] = ratio
    df["vol_expansion"] = (ratio > 1.0).astype(int)

    # Classify into regimes via rolling quantiles
    rolling_q33 = ratio.rolling(long_window, min_periods=long_window // 2).quantile(0.33)
    rolling_q66 = ratio.rolling(long_window, min_periods=long_window // 2).quantile(0.66)
    df["vol_regime_label"] = np.where(
        ratio <= rolling_q33, 0,
        np.where(ratio >= rolling_q66, 2, 1)
    )

    return df


def compute_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Overnight gap analysis.

    Gap = today's open / yesterday's close - 1
    Captures earnings surprises, overnight news impact.
    """
    if "open" not in df.columns or "close" not in df.columns:
        return df

    gap = df["open"] / df["close"].shift(1) - 1
    df["overnight_gap"] = gap
    df["overnight_gap_abs"] = gap.abs()
    df["gap_up"] = (gap > 0.005).astype(int)     # >0.5% gap up
    df["gap_down"] = (gap < -0.005).astype(int)   # >0.5% gap down

    # Gap fill tendency: does price close the gap during the day?
    # gap_fill_ratio = how much of the gap was closed
    intraday_move = df["close"] - df["open"]
    df["gap_fill_ratio"] = np.where(
        gap.abs() > 0.001,
        np.clip(-intraday_move / (gap * df["close"].shift(1)).replace(0, np.nan), -1, 2),
        0,
    )

    # Rolling gap statistics
    df["avg_gap_5d"] = gap.rolling(5).mean()
    df["gap_volatility_20d"] = gap.rolling(20).std()

    return df


def compute_volume_spike_features(
    df: pd.DataFrame,
    spike_multiplier: float = 2.5,
    window: int = 20,
) -> pd.DataFrame:
    """Volume spike detection and momentum.

    Volume spikes often precede or confirm price moves.
    """
    if "volume" not in df.columns:
        return df

    vol_ma = df["volume"].rolling(window).mean()
    vol_std = df["volume"].rolling(window).std()

    df["volume_zscore"] = (df["volume"] - vol_ma) / vol_std.replace(0, np.nan)
    df["volume_spike"] = (df["volume"] > vol_ma * spike_multiplier).astype(int)
    df["volume_trend_5d"] = (
        df["volume"].rolling(5).mean() / vol_ma
    ).replace([np.inf, -np.inf], np.nan)

    # Cumulative volume momentum
    df["volume_momentum_5d"] = (
        df["volume"].rolling(5).sum() /
        df["volume"].rolling(20).sum().replace(0, np.nan)
    )

    return df


def compute_correlation_features(
    df: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    window: int = 20,
) -> pd.DataFrame:
    """Rolling correlation with benchmark (if available).

    High correlation → stock follows the market.
    Divergence → stock-specific catalyst.
    """
    if benchmark_returns is None or "returns" not in df.columns:
        return df

    # Align
    aligned = pd.DataFrame({
        "asset": df["returns"],
        "benchmark": benchmark_returns,
    }).dropna()

    if len(aligned) < window:
        return df

    corr = aligned["asset"].rolling(window).corr(aligned["benchmark"])
    df["benchmark_corr_20d"] = corr.reindex(df.index)

    # Beta (rolling)
    cov = aligned["asset"].rolling(window).cov(aligned["benchmark"])
    var = aligned["benchmark"].rolling(window).var()
    df["rolling_beta_20d"] = (cov / var.replace(0, np.nan)).reindex(df.index)

    # Relative strength vs benchmark
    cum_asset = (1 + df["returns"]).rolling(window).apply(np.prod, raw=True) - 1
    cum_bench = (1 + benchmark_returns).rolling(window).apply(np.prod, raw=True) - 1
    df["relative_strength_20d"] = (cum_asset - cum_bench.reindex(df.index)).fillna(0)

    return df


# ────────────────────── Index-Specific Features ──────────────────────


def compute_index_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features unique to market indices.

    - Mean-reversion indicators (indices revert more than single stocks)
    - VIX-proxy features (realized vol as fear gauge)
    - Distance from 52-week high/low
    """
    # Mean-reversion: z-score on multiple windows
    if "close" in df.columns:
        for w in [10, 20, 50]:
            ma = df["close"].rolling(w).mean()
            std = df["close"].rolling(w).std()
            df[f"reversion_zscore_{w}d"] = (df["close"] - ma) / std.replace(0, np.nan)

        # Distance from 52-week high/low
        high_252 = df["close"].rolling(252, min_periods=50).max()
        low_252 = df["close"].rolling(252, min_periods=50).min()
        df["dist_52w_high"] = (df["close"] - high_252) / high_252
        df["dist_52w_low"] = (df["close"] - low_252) / low_252.replace(0, np.nan)

        # % of range (where are we in the 52-week range?)
        range_52w = high_252 - low_252
        df["pct_52w_range"] = (
            (df["close"] - low_252) / range_52w.replace(0, np.nan)
        )

    # VIX proxy: annualized realized volatility (similar to what VIX measures)
    if "returns" in df.columns:
        df["realized_vol_10d"] = df["returns"].rolling(10).std() * np.sqrt(252)
        df["realized_vol_30d"] = df["returns"].rolling(30).std() * np.sqrt(252)

        # Volatility term structure: short vol vs long vol
        df["vol_term_structure"] = (
            df["realized_vol_10d"] / df["realized_vol_30d"].replace(0, np.nan)
        )

        # Consecutive-days-in-range: how many days since last >1% move
        big_move = df["returns"].abs() > 0.01
        df["days_since_big_move"] = big_move.groupby(big_move.cumsum()).cumcount()

    return df


# ────────────────────── Main Dispatcher ──────────────────────


def compute_market_adaptive_features(
    df: pd.DataFrame,
    market_name: str,
    strategy_config: Optional[Dict] = None,
    benchmark_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute market-specific adaptive features.

    Dispatches to the appropriate feature generators based on
    the market type and adaptive config settings.

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with at least 'close', 'returns' columns.
    market_name : str
        Market identifier: 'stocks', 'crypto', 'futures', 'indices'.
    strategy_config : dict, optional
        Strategy config with 'market_adaptive' section.
    benchmark_returns : pd.Series, optional
        Benchmark return series for correlation features.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with market-adaptive features.
    """
    adaptive_cfg = {}
    if strategy_config:
        adaptive_cfg = strategy_config.get("market_adaptive", {})

    spike_mult = adaptive_cfg.get("volume_spike_multiplier", 2.5)
    n_features_before = len(df.columns)

    # ── Universal adaptive features ──
    if adaptive_cfg.get("volatility_regime", True):
        df = compute_volatility_regime(df)

    if adaptive_cfg.get("gap_analysis", False):
        df = compute_gap_features(df)

    df = compute_volume_spike_features(df, spike_multiplier=spike_mult)

    if adaptive_cfg.get("correlation_features", False):
        df = compute_correlation_features(df, benchmark_returns=benchmark_returns)

    # ── Market-specific features ──
    if market_name == "indices":
        df = compute_index_features(df)

    elif market_name == "stocks":
        # Stocks get gap analysis + index-style distance from highs
        if "close" in df.columns:
            high_252 = df["close"].rolling(252, min_periods=50).max()
            low_252 = df["close"].rolling(252, min_periods=50).min()
            df["dist_52w_high"] = (df["close"] - high_252) / high_252
            df["dist_52w_low"] = (df["close"] - low_252) / low_252.replace(0, np.nan)

    n_new = len(df.columns) - n_features_before
    logger.info(
        f"Market-adaptive features ({market_name}): "
        f"+{n_new} features → {len(df.columns)} total columns"
    )

    return df


def get_adaptive_feature_names(market_name: str) -> List[str]:
    """List the names of adaptive features for a given market.

    Useful for documentation and feature filtering.
    """
    # Universal
    universal = [
        "vol_regime_ratio", "vol_expansion", "vol_regime_label",
        "volume_zscore", "volume_spike", "volume_trend_5d", "volume_momentum_5d",
    ]

    gap = [
        "overnight_gap", "overnight_gap_abs", "gap_up", "gap_down",
        "gap_fill_ratio", "avg_gap_5d", "gap_volatility_20d",
    ]

    correlation = [
        "benchmark_corr_20d", "rolling_beta_20d", "relative_strength_20d",
    ]

    # Market-specific
    indices = [
        "reversion_zscore_10d", "reversion_zscore_20d", "reversion_zscore_50d",
        "dist_52w_high", "dist_52w_low", "pct_52w_range",
        "realized_vol_10d", "realized_vol_30d", "vol_term_structure",
        "days_since_big_move",
    ]

    stocks = ["dist_52w_high", "dist_52w_low"] + gap + correlation

    mapping = {
        "indices": universal + gap + indices + correlation,
        "stocks": universal + stocks,
    }

    return mapping.get(market_name, universal)
