"""
Return-based features.

Computes lagged returns, rolling statistics, and risk metrics.
All features are backward-looking (no look-ahead bias).
"""

import numpy as np
import pandas as pd


def compute_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all return-based features and append as new columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must have column 'close'. Ideally already has 'returns' and
        'log_returns' from preprocessing.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new return feature columns appended.
    """
    if df.empty:
        return df

    df = df.copy()

    # Ensure base returns exist
    if "returns" not in df.columns:
        df["returns"] = df["close"].pct_change()
    if "log_returns" not in df.columns:
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # ── Lagged Returns (past N-day returns) ──
    for n in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{n}d"] = df["close"].pct_change(periods=n)

    # ── Rolling Volatility (standard deviation of returns) ──
    for window in [5, 10, 20]:
        df[f"vol_{window}d"] = df["returns"].rolling(window=window).std()

    # ── Rolling Sharpe Ratio (annualized) ──
    # Sharpe = (mean_return / std_return) * sqrt(252)
    for window in [20, 60]:
        rolling_mean = df["returns"].rolling(window=window).mean()
        rolling_std = df["returns"].rolling(window=window).std()
        df[f"sharpe_{window}d"] = (
            (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)
        )

    # ── Rolling Max Drawdown ──
    for window in [20, 60]:
        df[f"max_dd_{window}d"] = _rolling_max_drawdown(df["close"], window)

    # ── Momentum (N-day return, same as ret_Nd but semantically distinct) ──
    # Momentum is the cumulative return over a window
    df["momentum_10d"] = df["close"] / df["close"].shift(10) - 1
    df["momentum_20d"] = df["close"] / df["close"].shift(20) - 1

    # ── Mean Reversion Signal ──
    # Distance from rolling mean as fraction of rolling std
    for window in [20, 50]:
        rolling_mean = df["close"].rolling(window=window).mean()
        rolling_std = df["close"].rolling(window=window).std()
        df[f"zscore_{window}d"] = (
            (df["close"] - rolling_mean) / rolling_std.replace(0, np.nan)
        )

    # ── Skewness and Kurtosis (rolling) ──
    df["skew_20d"] = df["returns"].rolling(window=20).skew()
    df["kurt_20d"] = df["returns"].rolling(window=20).kurt()

    # ── Up/Down day ratio (rolling) ──
    up = (df["returns"] > 0).astype(float)
    df["up_ratio_20d"] = up.rolling(window=20).mean()

    # ── Consecutive up/down days ──
    df["consec_up"] = _consecutive_sign(df["returns"], positive=True)
    df["consec_down"] = _consecutive_sign(df["returns"], positive=False)

    return df


def _rolling_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    """Compute rolling maximum drawdown over a given window.

    Max drawdown = (trough - peak) / peak over the window.
    Returns negative values (e.g., -0.10 = 10% drawdown).
    """
    result = pd.Series(index=close.index, dtype=float)

    for i in range(window, len(close)):
        window_data = close.iloc[i - window : i + 1]
        peak = window_data.expanding().max()
        drawdown = (window_data - peak) / peak
        result.iloc[i] = drawdown.min()

    return result


def _consecutive_sign(returns: pd.Series, positive: bool = True) -> pd.Series:
    """Count consecutive positive or negative return days.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    positive : bool
        If True, count consecutive positive days.
        If False, count consecutive negative days.

    Returns
    -------
    pd.Series
        Count of consecutive same-sign days ending at each row.
    """
    if positive:
        sign = (returns > 0).astype(int)
    else:
        sign = (returns < 0).astype(int)

    # Group consecutive same-sign runs
    groups = sign.ne(sign.shift()).cumsum()
    counts = sign.groupby(groups).cumsum()

    return counts


def get_return_feature_names() -> list:
    """Return the list of all return feature column names."""
    names = []

    # Lagged returns
    for n in [1, 2, 3, 5, 10, 20]:
        names.append(f"ret_{n}d")

    # Rolling volatility
    for w in [5, 10, 20]:
        names.append(f"vol_{w}d")

    # Rolling Sharpe
    for w in [20, 60]:
        names.append(f"sharpe_{w}d")

    # Rolling max drawdown
    for w in [20, 60]:
        names.append(f"max_dd_{w}d")

    # Momentum
    names.extend(["momentum_10d", "momentum_20d"])

    # Z-scores (mean reversion)
    for w in [20, 50]:
        names.append(f"zscore_{w}d")

    # Higher moments
    names.extend(["skew_20d", "kurt_20d"])

    # Up/down ratios
    names.extend(["up_ratio_20d", "consec_up", "consec_down"])

    return names
