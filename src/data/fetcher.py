"""
Data fetcher abstraction layer.

Provides a unified interface for fetching OHLCV data from multiple sources.
The YFinanceFetcher handles stocks, crypto, futures, and indices through a
single API with market-specific ticker formatting.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

from .market_config import MarketConfig

logger = logging.getLogger(__name__)


class DataFetcher(ABC):
    """Abstract base class for all data fetchers."""

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single ticker.

        Parameters
        ----------
        ticker : str
            The formatted ticker symbol (e.g. 'AAPL', 'BTC-USD', 'ES=F').
        start : str or datetime
            Start date for data retrieval.
        end : str or datetime
            End date for data retrieval.
        interval : str
            Data frequency. Default '1d' (daily).

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex and columns:
            [open, high, low, close, volume, adj_close]
        """
        pass

    @abstractmethod
    def fetch_multiple(
        self,
        tickers: List[str],
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple tickers.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Mapping of ticker -> DataFrame.
        """
        pass


class YFinanceFetcher(DataFetcher):
    """Fetcher using yfinance (Yahoo Finance unofficial API).

    Supports stocks, crypto, futures, and indices via unified yfinance API.
    Handles rate limiting, retries, and error recovery.
    """

    # Standardized output column names
    COLUMN_MAP = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adj_close",
    }

    OUTPUT_COLUMNS = ["open", "high", "low", "close", "volume", "adj_close"]

    def __init__(
        self,
        market_config: Optional[MarketConfig] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.market_config = market_config
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch(
        self,
        ticker: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single ticker with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                )

                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

                return self._standardize(df, ticker)

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed for {ticker}: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)  # exponential backoff
                else:
                    logger.error(f"All retries exhausted for {ticker}")
                    return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

        return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

    def fetch_multiple(
        self,
        tickers: List[str],
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers via batch download.

        Uses yfinance's batch download for efficiency, then splits
        the multi-level DataFrame into individual DataFrames.
        """
        result: Dict[str, pd.DataFrame] = {}

        if not tickers:
            return result

        logger.info(f"Fetching {len(tickers)} tickers from {start} to {end}")

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    group_by="ticker",
                    threads=True,
                )

                if raw.empty:
                    logger.warning("Batch download returned empty DataFrame")
                    return result

                # Single ticker: yf.download doesn't create multi-level columns
                if len(tickers) == 1:
                    df = self._standardize(raw, tickers[0])
                    if not df.empty:
                        result[tickers[0]] = df
                else:
                    # Multi-ticker: columns are (ticker, OHLCV)
                    for ticker in tickers:
                        try:
                            ticker_df = raw[ticker].copy()
                            df = self._standardize(ticker_df, ticker)
                            if not df.empty:
                                result[ticker] = df
                            else:
                                logger.warning(f"Empty data for {ticker}")
                        except KeyError:
                            logger.warning(f"Ticker {ticker} not found in batch result")

                logger.info(
                    f"Successfully fetched {len(result)}/{len(tickers)} tickers"
                )
                return result

            except Exception as e:
                logger.warning(
                    f"Batch attempt {attempt}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        # Fallback: fetch individually
        logger.info("Batch failed, falling back to individual fetches")
        for ticker in tickers:
            df = self.fetch(ticker, start, end, interval)
            if not df.empty:
                result[ticker] = df

        return result

    def _standardize(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Standardize a raw yfinance DataFrame to our unified schema.

        Output columns: [open, high, low, close, volume, adj_close]
        Index: DatetimeIndex (timezone-naive, date only for daily data)
        """
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

        # Handle multi-level columns from yfinance (ticker, price) format
        if isinstance(df.columns, pd.MultiIndex):
            # Drop the ticker level, keep only price columns
            df = df.droplevel("Ticker", axis=1)

        # Rename columns to lowercase standard
        df = df.rename(columns=self.COLUMN_MAP)

        # If 'adj_close' is missing (some markets), use 'close'
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        # Keep only standard columns that exist
        cols = [c for c in self.OUTPUT_COLUMNS if c in df.columns]
        df = df[cols].copy()

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Remove timezone info for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Sort by date
        df = df.sort_index()

        # Drop rows where all price columns are NaN
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        df = df.dropna(subset=price_cols, how="all")

        # Ensure numeric types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.debug(f"{ticker}: {len(df)} rows from {df.index[0]} to {df.index[-1]}")

        return df


def create_fetcher(
    market_config: Optional[MarketConfig] = None,
    source: str = "yfinance",
) -> DataFetcher:
    """Factory function to create the appropriate DataFetcher.

    Parameters
    ----------
    market_config : MarketConfig, optional
        Market configuration (for market-specific settings).
    source : str
        Data source name. Currently only 'yfinance' is supported.

    Returns
    -------
    DataFetcher
        An instance of the appropriate fetcher.
    """
    if source == "yfinance":
        return YFinanceFetcher(market_config=market_config)
    else:
        raise ValueError(
            f"Unknown data source: '{source}'. Available: ['yfinance']"
        )
