"""
Technical indicator features.

Computes trend, momentum, and volatility indicators using pandas-ta.
All functions are pure: input DataFrame in, new columns out.
Works across all markets (stocks, crypto, futures, indices).
"""

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


def compute_technical_indicators(
    df: pd.DataFrame,
    include_groups: list = None,
) -> pd.DataFrame:
    """Compute all technical indicators and append as new columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: [open, high, low, close, volume].
    include_groups : list, optional
        Which indicator groups to include. Options:
        ['trend', 'momentum', 'volatility', 'volume', 'custom']
        Default: all groups.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new indicator columns appended.
    """
    if df.empty:
        return df

    df = df.copy()

    all_groups = ["trend", "momentum", "volatility", "volume", "custom"]
    groups = include_groups or all_groups

    if "trend" in groups:
        df = _add_trend_indicators(df)

    if "momentum" in groups:
        df = _add_momentum_indicators(df)

    if "volatility" in groups:
        df = _add_volatility_indicators(df)

    if "volume" in groups:
        df = _add_volume_indicators(df)

    if "custom" in groups:
        df = _add_custom_features(df)

    # Count how many indicator columns we added
    n_new = len([c for c in df.columns if c not in [
        "open", "high", "low", "close", "volume", "adj_close",
        "returns", "log_returns", "volume_norm",
    ]])
    logger.debug(f"Added {n_new} technical indicator columns")

    return df


# ──────────────────── Indicator Groups ────────────────────


def _add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Trend indicators: SMA, EMA, MACD, ADX."""

    # Simple Moving Averages
    df["sma_20"] = ta.sma(df["close"], length=20)
    df["sma_50"] = ta.sma(df["close"], length=50)
    df["sma_200"] = ta.sma(df["close"], length=200)

    # Exponential Moving Averages
    df["ema_12"] = ta.ema(df["close"], length=12)
    df["ema_26"] = ta.ema(df["close"], length=26)

    # MACD (12, 26, 9)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["macd"] = macd.iloc[:, 0]         # MACD line
        df["macd_hist"] = macd.iloc[:, 1]    # MACD histogram
        df["macd_signal"] = macd.iloc[:, 2]  # Signal line

    # ADX (Average Directional Index) — trend strength
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx is not None and not adx.empty:
        df["adx"] = adx.iloc[:, 0]   # ADX
        df["dmp"] = adx.iloc[:, 1]   # +DI (Directional Movement Plus)
        df["dmn"] = adx.iloc[:, 2]   # -DI (Directional Movement Minus)

    return df


def _add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum indicators: RSI, Stochastic, CCI, Williams %R."""

    # RSI (Relative Strength Index)
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    # Stochastic Oscillator (14, 3)
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    if stoch is not None and not stoch.empty:
        df["stoch_k"] = stoch.iloc[:, 0]  # %K (fast)
        df["stoch_d"] = stoch.iloc[:, 1]  # %D (slow)

    # CCI (Commodity Channel Index)
    df["cci_20"] = ta.cci(df["high"], df["low"], df["close"], length=20)

    # Williams %R
    df["willr_14"] = ta.willr(df["high"], df["low"], df["close"], length=14)

    return df


def _add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility indicators: Bollinger Bands, ATR, Keltner Channels."""

    # Bollinger Bands (20, 2)
    bbands = ta.bbands(df["close"], length=20, std=2)
    if bbands is not None and not bbands.empty:
        df["bb_lower"] = bbands.iloc[:, 0]   # Lower band
        df["bb_mid"] = bbands.iloc[:, 1]     # Middle band (SMA 20)
        df["bb_upper"] = bbands.iloc[:, 2]   # Upper band
        # Bandwidth and %B
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            bb_width = df["bb_upper"] - df["bb_lower"]
            df["bb_width"] = bb_width / df["bb_mid"]
            df["bb_pct"] = (df["close"] - df["bb_lower"]) / bb_width.replace(0, np.nan)

    # ATR (Average True Range)
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # ATR as % of price (normalized — comparable across assets)
    if "atr_14" in df.columns:
        df["atr_pct"] = df["atr_14"] / df["close"]

    return df


def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Volume indicators: OBV, volume SMA ratio."""

    if "volume" not in df.columns:
        return df

    # OBV (On-Balance Volume)
    df["obv"] = ta.obv(df["close"], df["volume"])

    # Volume relative to 20-day average
    vol_sma = ta.sma(df["volume"], length=20)
    if vol_sma is not None:
        df["vol_ratio_20"] = df["volume"] / vol_sma.replace(0, np.nan)

    return df


def _add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Custom derived features — trading signals and cross-indicator features."""

    # SMA crossover signals (1 if short > long, 0 otherwise)
    if "sma_20" in df.columns and "sma_50" in df.columns:
        df["sma_cross_20_50"] = (df["sma_20"] > df["sma_50"]).astype(int)

    if "sma_50" in df.columns and "sma_200" in df.columns:
        df["sma_cross_50_200"] = (df["sma_50"] > df["sma_200"]).astype(int)

    # Distance from key moving averages (as % of price)
    if "sma_20" in df.columns:
        df["dist_sma_20"] = (df["close"] - df["sma_20"]) / df["sma_20"]

    if "sma_50" in df.columns:
        df["dist_sma_50"] = (df["close"] - df["sma_50"]) / df["sma_50"]

    if "sma_200" in df.columns:
        df["dist_sma_200"] = (df["close"] - df["sma_200"]) / df["sma_200"]

    # RSI zones (oversold < 30, overbought > 70)
    if "rsi_14" in df.columns:
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)

    # MACD crossover (MACD line > signal line)
    if "macd" in df.columns and "macd_signal" in df.columns:
        df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int)

    # Trend strength bins based on ADX
    if "adx" in df.columns:
        df["trend_strong"] = (df["adx"] > 25).astype(int)

    # Price position within daily range: (close - low) / (high - low)
    daily_range = df["high"] - df["low"]
    df["close_position"] = (df["close"] - df["low"]) / daily_range.replace(0, np.nan)

    return df


def get_feature_names() -> list:
    """Return the list of all feature column names this module generates.

    Useful for documentation and feature selection.
    """
    return [
        # Trend
        "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
        "macd", "macd_hist", "macd_signal",
        "adx", "dmp", "dmn",
        # Momentum
        "rsi_14", "stoch_k", "stoch_d", "cci_20", "willr_14",
        # Volatility
        "bb_lower", "bb_mid", "bb_upper", "bb_width", "bb_pct",
        "atr_14", "atr_pct",
        # Volume
        "obv", "vol_ratio_20",
        # Custom
        "sma_cross_20_50", "sma_cross_50_200",
        "dist_sma_20", "dist_sma_50", "dist_sma_200",
        "rsi_oversold", "rsi_overbought", "macd_cross",
        "trend_strong", "close_position",
    ]
