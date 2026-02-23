"""
Market Regime Detector for MarketPulse Phase 3.

Identifies market regimes using a rules-based approach augmented by
statistical clustering.  Three primary regimes are detected:

    0 = BEARISH  – downtrend, high/rising vol
    1 = NEUTRAL  – range-bound, moderate vol
    2 = BULLISH  – uptrend, low/falling vol

The detector runs as a **labelling** step (not a predictive model) so
regime labels are computed from past data only—no look-ahead bias.

Main public API
---------------
- ``MarketRegimeDetector``  — configurable class
- ``detect_regime()``       — quick functional interface
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Regime Constants ──

BEARISH = 0
NEUTRAL = 1
BULLISH = 2
REGIME_LABELS = {BEARISH: "bearish", NEUTRAL: "neutral", BULLISH: "bullish"}


# ── Config ──

@dataclass
class RegimeConfig:
    """Parameters for the rules-based regime detector."""

    # Trend detection
    trend_short_window: int = 20
    trend_long_window: int = 50
    trend_ema: bool = True  # Use EMA for trend, else SMA

    # Volatility detection
    vol_window: int = 20
    vol_long_window: int = 60

    # ADX-like trend strength
    adx_window: int = 14
    adx_threshold_trending: float = 25.0  # ADX > 25 → trending

    # Thresholds
    bullish_return_threshold: float = 0.0   # short MA above long MA by this
    bearish_return_threshold: float = 0.0   # short MA below long MA by this

    # Smoothing (majority vote over last N days)
    smooth_window: int = 5

    @classmethod
    def for_market(cls, market_name: str) -> "RegimeConfig":
        """Return sensible defaults per market type."""
        configs = {
            "crypto": cls(
                trend_short_window=10,
                trend_long_window=30,
                vol_window=10,
                vol_long_window=30,
                smooth_window=3,
            ),
            "futures": cls(
                trend_short_window=15,
                trend_long_window=40,
                vol_window=15,
                vol_long_window=50,
                smooth_window=5,
            ),
            "indices": cls(
                trend_short_window=20,
                trend_long_window=50,
                vol_window=20,
                vol_long_window=60,
                smooth_window=7,
            ),
            "stocks": cls(),  # defaults
        }
        return configs.get(market_name, cls())


# ── Detector ──

class MarketRegimeDetector:
    """Rules-based market regime detector.

    Algorithm:
    1. Compute trend direction  (short MA vs long MA).
    2. Compute trend strength   (simplified ADX proxy).
    3. Compute volatility regime (short vol vs long vol).
    4. Combine signals into raw regime score.
    5. Smooth with majority-vote window to avoid whipsaws.

    No look-ahead: all signals use expanding / trailing windows only.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()

    # ── public ──

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime columns to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have 'close' column. Optionally 'high', 'low', 'returns'.

        Returns
        -------
        pd.DataFrame
            Enriched with:
            - regime: int (0/1/2)
            - regime_label: str (bearish/neutral/bullish)
            - regime_score: float (-1 to +1, raw signal before discretisation)
            - trend_direction: float (-1 to +1)
            - vol_regime: float (short_vol / long_vol)
            - trend_strength: float (0 to 1, ADX-proxy normalised)
        """
        if "close" not in df.columns:
            logger.warning("regime_detect: 'close' column missing — skipping")
            return df

        df = self._add_trend_direction(df)
        df = self._add_trend_strength(df)
        df = self._add_vol_regime(df)
        df = self._combine_regime(df)
        df = self._smooth_regime(df)
        return df

    def get_regime_summary(self, df: pd.DataFrame) -> Dict:
        """Return summary statistics of regime distribution."""
        if "regime" not in df.columns:
            return {}

        regime_col = df["regime"].dropna()
        total = len(regime_col)

        summary = {
            "regime_counts": regime_col.value_counts().to_dict(),
            "regime_pcts": (regime_col.value_counts() / total * 100).round(1).to_dict(),
            "current_regime": REGIME_LABELS.get(int(regime_col.iloc[-1]), "unknown"),
            "regime_changes": int((regime_col != regime_col.shift()).sum() - 1),
            "avg_regime_duration_days": round(total / max((regime_col != regime_col.shift()).sum(), 1), 1),
        }
        return summary

    # ── private ──

    def _moving_average(self, s: pd.Series, window: int) -> pd.Series:
        if self.config.trend_ema:
            return s.ewm(span=window, adjust=False).mean()
        return s.rolling(window).mean()

    def _add_trend_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Short MA vs Long MA → trend direction in [-1, +1]."""
        c = self.config
        short_ma = self._moving_average(df["close"], c.trend_short_window)
        long_ma = self._moving_average(df["close"], c.trend_long_window)

        # Normalised difference: (short - long) / long
        diff = (short_ma - long_ma) / long_ma.replace(0, np.nan)

        # Clip to [-1, 1] — the raw diff is usually small (e.g., ±0.05)
        # so we scale by 20× to get a broader spread, then clip
        df["trend_direction"] = np.clip(diff * 20, -1, 1)
        return df

    def _add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """ADX-proxy: average absolute return over window, normalised to [0, 1]."""
        c = self.config

        if "returns" not in df.columns:
            df["trend_strength"] = np.nan
            return df

        # Directional movement: absolute mean return / mean absolute return
        # If price consistently moves in one direction, this → 1
        # If price is choppy, this → 0
        window = c.adx_window
        mean_ret = df["returns"].rolling(window).mean()
        mean_abs = df["returns"].abs().rolling(window).mean()

        raw_strength = (mean_ret.abs() / mean_abs.replace(0, np.nan)).fillna(0)
        df["trend_strength"] = np.clip(raw_strength, 0, 1)
        return df

    def _add_vol_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Short vol / long vol ratio."""
        c = self.config

        if "returns" not in df.columns:
            df["vol_regime"] = np.nan
            return df

        short_vol = df["returns"].rolling(c.vol_window).std()
        long_vol = df["returns"].rolling(c.vol_long_window).std()
        df["vol_regime"] = short_vol / long_vol.replace(0, np.nan)
        return df

    def _combine_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine signals into regime_score and raw_regime.

        Regime score formula:
            score = trend_direction × trend_strength − vol_penalty

        Where vol_penalty pushes toward NEUTRAL when vol is expanding.
        """
        trend_dir = df.get("trend_direction", pd.Series(0, index=df.index))
        trend_str = df.get("trend_strength", pd.Series(0, index=df.index))
        vol_regime = df.get("vol_regime", pd.Series(1, index=df.index))

        # Vol penalty: high vol expansion reduces confidence in trend
        vol_penalty = np.clip((vol_regime - 1.0) * 0.3, -0.2, 0.3)

        score = trend_dir * trend_str - vol_penalty
        df["regime_score"] = np.clip(score, -1, 1)

        # Discretise
        df["raw_regime"] = np.where(
            score > 0.15, BULLISH,
            np.where(score < -0.15, BEARISH, NEUTRAL)
        )
        return df

    def _smooth_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Majority-vote smoothing to reduce regime flip-flop."""
        window = self.config.smooth_window
        raw = df["raw_regime"]

        if window <= 1:
            df["regime"] = raw
        else:
            # Rolling mode (majority vote)
            df["regime"] = raw.rolling(window, min_periods=1).apply(
                lambda x: pd.Series(x).mode().iloc[0], raw=False
            ).astype(int)

        df["regime_label"] = df["regime"].map(REGIME_LABELS)

        # Clean up
        df.drop(columns=["raw_regime"], inplace=True, errors="ignore")
        return df


# ── Functional convenience API ──


def detect_regime(
    df: pd.DataFrame,
    market_name: str = "stocks",
    config: Optional[RegimeConfig] = None,
) -> pd.DataFrame:
    """Detect market regime with sensible per-market defaults.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'close' (+ 'returns' recommended).
    market_name : str
        One of 'stocks', 'crypto', 'futures', 'indices'.
    config : RegimeConfig, optional
        Override auto-detected config.

    Returns
    -------
    pd.DataFrame
        Enriched with regime columns.
    """
    if config is None:
        config = RegimeConfig.for_market(market_name)

    detector = MarketRegimeDetector(config)
    return detector.detect(df)


def get_regime_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where the regime changed.

    Useful for detecting regime-shift dates.
    """
    if "regime" not in df.columns:
        return pd.DataFrame()

    mask = df["regime"] != df["regime"].shift()
    transitions = df[mask].copy()
    transitions["prev_regime"] = df["regime"].shift()[mask]
    return transitions
