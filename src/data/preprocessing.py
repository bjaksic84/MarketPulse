"""
Data preprocessing pipeline.

Cleans, aligns, and normalizes raw OHLCV data from any market.
Handles missing data, volume normalization, and calendar alignment.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .market_config import MarketConfig

logger = logging.getLogger(__name__)


def preprocess_ohlcv(
    df: pd.DataFrame,
    market_config: Optional[MarketConfig] = None,
    max_missing_pct: float = 0.05,
    fill_method: str = "ffill",
    max_gap_days: int = 3,
) -> pd.DataFrame:
    """Preprocess a single ticker's OHLCV data.

    Steps:
    1. Drop duplicate dates
    2. Forward-fill small gaps (≤ max_gap_days)
    3. Validate data quality (reject if > max_missing_pct missing)
    4. Ensure adjusted close is used when applicable
    5. Normalize volume
    6. Add basic derived columns (returns, log_returns)

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data with DatetimeIndex.
    market_config : MarketConfig, optional
        Market-specific configuration.
    max_missing_pct : float
        Max fraction of missing values allowed (default 5%).
    fill_method : str
        Method for filling gaps: 'ffill' (forward fill).
    max_gap_days : int
        Maximum consecutive gap days to fill (default 3).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with added columns: [returns, log_returns, volume_norm]
    """
    if df.empty:
        return df

    df = df.copy()

    # 1. Drop duplicate dates, keep last
    df = df[~df.index.duplicated(keep="last")]

    # 2. Sort by date
    df = df.sort_index()

    # 3. Check data quality before filling
    total_expected = len(pd.bdate_range(df.index[0], df.index[-1]))
    if total_expected > 0:
        missing_pct = 1 - (len(df) / total_expected)
        if missing_pct > max_missing_pct:
            logger.warning(
                f"Data has {missing_pct:.1%} missing values "
                f"(threshold: {max_missing_pct:.1%}). Proceeding with caution."
            )

    # 4. Forward-fill small gaps in price data
    price_cols = ["open", "high", "low", "close", "adj_close"]
    existing_price_cols = [c for c in price_cols if c in df.columns]
    df[existing_price_cols] = df[existing_price_cols].ffill(limit=max_gap_days)

    # Fill volume gaps with 0 (no trading = 0 volume)
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)

    # 5. Use adjusted close for the 'close' column if applicable
    use_adj = (
        market_config.use_adjusted_close if market_config else True
    )
    if use_adj and "adj_close" in df.columns:
        # Calculate adjustment ratio
        if "close" in df.columns:
            adj_ratio = df["adj_close"] / df["close"]
            # Adjust OHLC
            for col in ["open", "high", "low"]:
                if col in df.columns:
                    df[col] = df[col] * adj_ratio
            df["close"] = df["adj_close"]

    # 6. Drop any remaining rows with NaN in close
    df = df.dropna(subset=["close"])

    # 7. Compute basic returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # 8. Normalize volume
    vol_method = (
        market_config.volume_normalization if market_config else "z_score"
    )
    if "volume" in df.columns and len(df) > 1:
        df["volume_norm"] = _normalize_volume(df["volume"], method=vol_method)
    else:
        df["volume_norm"] = 0.0

    # 9. Drop the first row (NaN from returns calculation)
    df = df.iloc[1:]

    return df


def preprocess_multiple(
    data: Dict[str, pd.DataFrame],
    market_config: Optional[MarketConfig] = None,
    max_missing_pct: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """Preprocess multiple tickers, dropping those that fail quality checks.

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Mapping of ticker -> raw OHLCV DataFrame.
    market_config : MarketConfig, optional
        Market-specific configuration.
    max_missing_pct : float
        Max fraction of missing values allowed.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping of ticker -> preprocessed DataFrame (only valid tickers).
    """
    result = {}
    for ticker, df in data.items():
        try:
            processed = preprocess_ohlcv(
                df, market_config=market_config, max_missing_pct=max_missing_pct
            )
            if not processed.empty and len(processed) > 50:
                result[ticker] = processed
                logger.debug(f"{ticker}: {len(processed)} clean rows")
            else:
                logger.warning(f"{ticker}: insufficient data after preprocessing")
        except Exception as e:
            logger.warning(f"{ticker}: preprocessing failed: {e}")

    logger.info(
        f"Preprocessed {len(result)}/{len(data)} tickers successfully"
    )
    return result


def align_to_common_dates(
    data: Dict[str, pd.DataFrame],
    method: str = "inner",
) -> Dict[str, pd.DataFrame]:
    """Align multiple tickers to a common date index.

    Useful when combining features across tickers or markets.

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Mapping of ticker -> preprocessed DataFrame.
    method : str
        'inner' = keep only dates where ALL tickers have data.
        'outer' = keep all dates, fill missing with NaN.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Aligned DataFrames.
    """
    if not data:
        return data

    # Compute the common index
    all_indices = [df.index for df in data.values()]

    if method == "inner":
        common_idx = all_indices[0]
        for idx in all_indices[1:]:
            common_idx = common_idx.intersection(idx)
    else:  # outer
        common_idx = all_indices[0]
        for idx in all_indices[1:]:
            common_idx = common_idx.union(idx)

    common_idx = common_idx.sort_values()

    return {
        ticker: df.reindex(common_idx) for ticker, df in data.items()
    }


def _normalize_volume(volume: pd.Series, method: str = "z_score") -> pd.Series:
    """Normalize volume using the specified method.

    Parameters
    ----------
    volume : pd.Series
        Raw volume data.
    method : str
        Normalization method: 'z_score', 'min_max', or 'log'.

    Returns
    -------
    pd.Series
        Normalized volume.
    """
    if method == "z_score":
        mean = volume.rolling(window=50, min_periods=20).mean()
        std = volume.rolling(window=50, min_periods=20).std()
        std = std.replace(0, 1)  # avoid division by zero
        return (volume - mean) / std

    elif method == "min_max":
        rolling_min = volume.rolling(window=50, min_periods=20).min()
        rolling_max = volume.rolling(window=50, min_periods=20).max()
        denom = rolling_max - rolling_min
        denom = denom.replace(0, 1)
        return (volume - rolling_min) / denom

    elif method == "log":
        return np.log1p(volume)

    else:
        raise ValueError(f"Unknown volume normalization method: '{method}'")
