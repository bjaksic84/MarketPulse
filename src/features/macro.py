"""
Macro & calendar features for MarketPulse Phase 4.

Provides market-context features that technical indicators cannot capture:

1. **Calendar effects**: month-of-year, day-of-week, quarter-end,
   month-end, options expiry proxy (3rd Friday)
2. **VIX proxy**: realized volatility as a fear gauge
3. **Macro regime signals**: yield curve slope proxy, risk-on/off
4. **Trend context**: long-term returns & momentum state

These features address the key weakness of pure-technical models:
they give the model awareness of WHERE we are in macro/calendar cycles.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────── Calendar Features ──────────────────────


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-based features that capture seasonal/cyclical patterns.

    These features are powerful for indices and futures where
    institutional behavior follows calendar patterns.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Calendar features require DatetimeIndex")
        return df

    idx = df.index

    # Day of week (normalised 0-1)
    df["cal_day_of_week"] = idx.dayofweek / 4.0  # 0=Mon, 1=Fri

    # Month of year (normalised, cyclical encoding)
    month = idx.month
    df["cal_month_sin"] = np.sin(2 * np.pi * month / 12)
    df["cal_month_cos"] = np.cos(2 * np.pi * month / 12)

    # Quarter
    df["cal_quarter"] = idx.quarter / 4.0

    # Day-of-month normalised (for month-end effects)
    df["cal_day_of_month"] = idx.day / idx.days_in_month

    # Month-end proximity (last 5 trading days of month)
    df["cal_month_end"] = (idx.day >= (idx.days_in_month - 5)).astype(int)

    # Quarter-end proximity (last 10 trading days of quarter)
    df["cal_quarter_end"] = (
        (idx.month.isin([3, 6, 9, 12])) & (idx.day >= 15)
    ).astype(int)

    # Year-end proximity
    df["cal_year_end"] = (
        (idx.month == 12) & (idx.day >= 15)
    ).astype(int)

    # January effect (stocks tend to rally in January)
    df["cal_january"] = (idx.month == 1).astype(int)

    # Options expiry proxy: 3rd Friday of the month
    # Monthly options expire on 3rd Friday → high volume, gamma exposure
    third_friday = _get_third_friday_mask(idx)
    df["cal_opex_week"] = third_friday.astype(int)

    # Monday/Friday effects
    df["cal_is_monday"] = (idx.dayofweek == 0).astype(int)
    df["cal_is_friday"] = (idx.dayofweek == 4).astype(int)

    return df


def _get_third_friday_mask(dates: pd.DatetimeIndex) -> pd.Series:
    """Identify the week of the 3rd Friday (options expiry week)."""
    # 3rd Friday = first Friday on or after the 15th of the month
    is_opex_week = pd.Series(False, index=dates)

    for date in dates:
        # Find the 3rd Friday of this month
        first_day = date.replace(day=1)
        # Days until Friday from the 1st
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day.day + days_to_friday
        third_friday_day = first_friday + 14

        # Is current date within 2 days of 3rd Friday?
        if abs(date.day - third_friday_day) <= 2:
            is_opex_week[date] = True

    return is_opex_week


# ────────────────────── VIX Proxy Features ──────────────────────


def compute_vix_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """VIX-proxy features from realized volatility.

    Since we may not have actual VIX data for all markets,
    we construct a realized-vol-based fear gauge.
    """
    if "returns" not in df.columns:
        return df

    # Realized volatility at multiple horizons (annualised)
    for window in [5, 10, 20]:
        vol = df["returns"].rolling(window).std() * np.sqrt(252)
        df[f"vix_proxy_{window}d"] = vol

    # VIX term structure: short vol vs long vol
    short_vol = df["returns"].rolling(5).std()
    long_vol = df["returns"].rolling(30).std()
    df["vix_term_structure"] = short_vol / long_vol.replace(0, np.nan)

    # VIX regime: percentile of current vol in historical context
    vol_20 = df["returns"].rolling(20).std()
    df["vix_percentile"] = vol_20.rolling(252, min_periods=60).apply(
        lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else 0.5,
        raw=False,
    )

    # Fear spike: vol expansion rate
    df["vix_spike"] = (vol_20 / vol_20.shift(5) - 1).clip(-1, 3)

    # Calm vs fearful regime
    df["vix_high_fear"] = (df["vix_percentile"] > 0.8).astype(int)
    df["vix_low_fear"] = (df["vix_percentile"] < 0.2).astype(int)

    return df


# ────────────────────── Trend Context Features ──────────────────────


def compute_trend_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Long-term trend context that short-term features miss.

    These features tell the model whether we're in a bull market,
    bear market, or recovery — crucial context for medium-term prediction.
    """
    if "close" not in df.columns:
        return df

    # Distance from key moving averages
    for window in [50, 100, 200]:
        ma = df["close"].rolling(window).mean()
        df[f"dist_ma_{window}d"] = (df["close"] - ma) / ma.replace(0, np.nan)

    # Golden/death cross: 50-day MA vs 200-day MA
    ma_50 = df["close"].rolling(50).mean()
    ma_200 = df["close"].rolling(200).mean()
    df["golden_cross"] = (ma_50 > ma_200).astype(int)

    # Cumulative return at multiple horizons
    if "returns" in df.columns:
        for window in [20, 60, 120]:
            df[f"cum_return_{window}d"] = (
                (1 + df["returns"]).rolling(window).apply(np.prod, raw=True) - 1
            )

    # Drawdown from rolling high
    rolling_max = df["close"].expanding().max()
    df["drawdown"] = (df["close"] - rolling_max) / rolling_max

    # Recovery: drawdown is improving (less negative)
    df["drawdown_recovery"] = (df["drawdown"] > df["drawdown"].shift(5)).astype(int)

    # Higher-highs / lower-lows (trend confirmation)
    if "high" in df.columns and "low" in df.columns:
        df["higher_high_20d"] = (
            df["high"] >= df["high"].rolling(20).max().shift(1)
        ).astype(int)
        df["lower_low_20d"] = (
            df["low"] <= df["low"].rolling(20).min().shift(1)
        ).astype(int)

    return df


# ────────────────────── Cross-Asset Context ──────────────────────


def compute_cross_asset_features(
    df: pd.DataFrame,
    vix_data: Optional[pd.Series] = None,
    benchmark_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Cross-asset context features.

    If actual VIX or benchmark data is provided, use it.
    Otherwise, skip gracefully.
    """
    # Real VIX data if available
    if vix_data is not None:
        aligned_vix = vix_data.reindex(df.index, method="ffill")
        df["real_vix"] = aligned_vix
        df["real_vix_change"] = aligned_vix.pct_change()
        df["real_vix_high"] = (aligned_vix > 25).astype(int)
        df["real_vix_extreme"] = (aligned_vix > 35).astype(int)

    # Relative performance vs benchmark
    if benchmark_data is not None and "returns" in df.columns:
        bench_ret = benchmark_data.get("returns")
        if bench_ret is not None:
            aligned_bench = bench_ret.reindex(df.index, method="ffill")
            df["relative_return_1d"] = df["returns"] - aligned_bench
            df["relative_return_20d"] = (
                df["returns"].rolling(20).sum() -
                aligned_bench.rolling(20).sum()
            )

    return df


# ────────────────────── Main Dispatcher ──────────────────────


def compute_macro_features(
    df: pd.DataFrame,
    strategy_config: Optional[Dict] = None,
    vix_data: Optional[pd.Series] = None,
    benchmark_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute all macro/calendar features.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'close', 'returns', DatetimeIndex.
    strategy_config : dict, optional
        Strategy config (for future feature toggles).
    vix_data : pd.Series, optional
        Actual VIX close data, if available.
    benchmark_data : pd.DataFrame, optional
        Benchmark OHLCV data for relative features.

    Returns
    -------
    pd.DataFrame
        Enriched with macro features.
    """
    n_before = len(df.columns)

    df = compute_calendar_features(df)
    df = compute_vix_proxy_features(df)
    df = compute_trend_context_features(df)
    df = compute_cross_asset_features(df, vix_data=vix_data, benchmark_data=benchmark_data)

    n_new = len(df.columns) - n_before
    logger.info(f"Macro features: +{n_new} features → {len(df.columns)} total columns")

    return df


def get_macro_feature_names() -> list:
    """List all possible macro feature names."""
    return [
        # Calendar
        "cal_day_of_week", "cal_month_sin", "cal_month_cos",
        "cal_quarter", "cal_day_of_month", "cal_month_end",
        "cal_quarter_end", "cal_year_end", "cal_january",
        "cal_opex_week", "cal_is_monday", "cal_is_friday",
        # VIX proxy
        "vix_proxy_5d", "vix_proxy_10d", "vix_proxy_20d",
        "vix_term_structure", "vix_percentile", "vix_spike",
        "vix_high_fear", "vix_low_fear",
        # Trend context
        "dist_ma_50d", "dist_ma_100d", "dist_ma_200d",
        "golden_cross", "cum_return_20d", "cum_return_60d",
        "cum_return_120d", "drawdown", "drawdown_recovery",
        "higher_high_20d", "lower_low_20d",
        # Cross-asset (optional)
        "real_vix", "real_vix_change", "real_vix_high", "real_vix_extreme",
        "relative_return_1d", "relative_return_20d",
    ]
